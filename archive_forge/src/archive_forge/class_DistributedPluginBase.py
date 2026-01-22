import sys
from copy import deepcopy
from glob import glob
import os
import shutil
from time import sleep, time
from traceback import format_exception
import numpy as np
from ... import logging
from ...utils.misc import str2bool
from ..engine.utils import topological_sort, load_resultfile
from ..engine import MapNode
from .tools import report_crash, report_nodes_not_run, create_pyscript
class DistributedPluginBase(PluginBase):
    """
    Execute workflow with a distribution engine

    Combinations of ``proc_done`` and ``proc_pending``:

    +------------+---------------+--------------------------------+
    | proc_done  | proc_pending  | outcome                        |
    +============+===============+================================+
    | True       | False         | Process is finished            |
    +------------+---------------+--------------------------------+
    | True       | True          | Process is currently being run |
    +------------+---------------+--------------------------------+
    | False      | False         | Process is queued              |
    +------------+---------------+--------------------------------+
    | False      | True          | INVALID COMBINATION            |
    +------------+---------------+--------------------------------+

    Attributes
    ----------
    procs : :obj:`list`
        list (N) of underlying interface elements to be processed
    proc_done : :obj:`numpy.ndarray`
        a boolean numpy array (N,) signifying whether a process has been
        submitted for execution
    proc_pending : :obj:`numpy.ndarray`
        a boolean numpy array (N,) signifying whether a
        process is currently running.
    depidx : :obj:`numpy.matrix`
        a boolean matrix (NxN) storing the dependency structure across
        processes. Process dependencies are derived from each column.

    """

    def __init__(self, plugin_args=None):
        """
        Initialize runtime attributes to none

        """
        super(DistributedPluginBase, self).__init__(plugin_args=plugin_args)
        self.procs = None
        self.depidx = None
        self.refidx = None
        self.mapnodes = None
        self.mapnodesubids = None
        self.proc_done = None
        self.proc_pending = None
        self.pending_tasks = []
        self.max_jobs = self.plugin_args.get('max_jobs', np.inf)

    def _prerun_check(self, graph):
        """Stub method to validate/massage graph and nodes before running"""

    def _postrun_check(self):
        """Stub method to close any open resources"""

    def run(self, graph, config, updatehash=False):
        """
        Executes a pre-defined pipeline using distributed approaches
        """
        logger.info('Running in parallel.')
        self._config = config
        poll_sleep_secs = float(config['execution']['poll_sleep_duration'])
        self._prerun_check(graph)
        self._generate_dependency_list(graph)
        self.mapnodes = []
        self.mapnodesubids = {}
        notrun = []
        errors = []
        old_progress_stats = None
        old_presub_stats = None
        while not np.all(self.proc_done) or np.any(self.proc_pending):
            loop_start = time()
            jobs_ready = np.nonzero(~self.proc_done & (self.depidx.sum(0) == 0))[1]
            progress_stats = (len(self.proc_done), np.sum(self.proc_done ^ self.proc_pending), np.sum(self.proc_done & self.proc_pending), len(jobs_ready), len(self.pending_tasks), np.sum(~self.proc_done & ~self.proc_pending))
            display_stats = progress_stats != old_progress_stats
            if display_stats:
                logger.debug('Progress: %d jobs, %d/%d/%d (done/running/ready), %d/%d (pending_tasks/waiting).', *progress_stats)
                old_progress_stats = progress_stats
            toappend = []
            while self.pending_tasks:
                taskid, jobid = self.pending_tasks.pop()
                try:
                    result = self._get_result(taskid)
                except Exception as exc:
                    notrun.append(self._clean_queue(jobid, graph))
                    errors.append(exc)
                else:
                    if result:
                        if result['traceback']:
                            notrun.append(self._clean_queue(jobid, graph, result=result))
                            errors.append(''.join(result['traceback']))
                        else:
                            self._task_finished_cb(jobid)
                            self._remove_node_dirs()
                        self._clear_task(taskid)
                    else:
                        assert self.proc_done[jobid] and self.proc_pending[jobid]
                        toappend.insert(0, (taskid, jobid))
            if toappend:
                self.pending_tasks.extend(toappend)
            num_jobs = len(self.pending_tasks)
            presub_stats = (num_jobs, np.sum(self.proc_done & self.proc_pending))
            display_stats = display_stats or presub_stats != old_presub_stats
            if display_stats:
                logger.debug('Tasks currently running: %d. Pending: %d.', *presub_stats)
                old_presub_stats = presub_stats
            if num_jobs < self.max_jobs:
                self._send_procs_to_workers(updatehash=updatehash, graph=graph)
            elif display_stats:
                logger.debug('Not submitting (max jobs reached)')
            sleep_til = loop_start + poll_sleep_secs
            sleep(max(0, sleep_til - time()))
        self._remove_node_dirs()
        report_nodes_not_run(notrun)
        self._postrun_check()
        if errors:
            error, cause = (errors[0], None)
            if isinstance(error, str):
                error = RuntimeError(error)
            if len(errors) > 1:
                error, cause = (RuntimeError(f'{len(errors)} raised. Re-raising first.'), error)
            raise error from cause

    def _get_result(self, taskid):
        raise NotImplementedError

    def _submit_job(self, node, updatehash=False):
        raise NotImplementedError

    def _report_crash(self, node, result=None):
        tb = None
        if result is not None:
            node._result = result['result']
            tb = result['traceback']
            node._traceback = tb
        return report_crash(node, traceback=tb)

    def _clear_task(self, taskid):
        raise NotImplementedError

    def _clean_queue(self, jobid, graph, result=None):
        logger.debug('Clearing %d from queue', jobid)
        if self._status_callback:
            self._status_callback(self.procs[jobid], 'exception')
        if result is None:
            result = {'result': None, 'traceback': '\n'.join(format_exception(*sys.exc_info()))}
        crashfile = self._report_crash(self.procs[jobid], result=result)
        if str2bool(self._config['execution']['stop_on_first_crash']):
            raise RuntimeError(''.join(result['traceback']))
        if jobid in self.mapnodesubids:
            self.proc_pending[jobid] = False
            self.proc_done[jobid] = True
            jobid = self.mapnodesubids[jobid]
            self.proc_pending[jobid] = False
            self.proc_done[jobid] = True
        return self._remove_node_deps(jobid, crashfile, graph)

    def _submit_mapnode(self, jobid):
        import scipy.sparse as ssp
        if jobid in self.mapnodes:
            return True
        self.mapnodes.append(jobid)
        mapnodesubids = self.procs[jobid].get_subnodes()
        numnodes = len(mapnodesubids)
        logger.debug('Adding %d jobs for mapnode %s', numnodes, self.procs[jobid])
        for i in range(numnodes):
            self.mapnodesubids[self.depidx.shape[0] + i] = jobid
        self.procs.extend(mapnodesubids)
        self.depidx = ssp.vstack((self.depidx, ssp.lil_matrix(np.zeros((numnodes, self.depidx.shape[1])))), 'lil')
        self.depidx = ssp.hstack((self.depidx, ssp.lil_matrix(np.zeros((self.depidx.shape[0], numnodes)))), 'lil')
        self.depidx[-numnodes:, jobid] = 1
        self.proc_done = np.concatenate((self.proc_done, np.zeros(numnodes, dtype=bool)))
        self.proc_pending = np.concatenate((self.proc_pending, np.zeros(numnodes, dtype=bool)))
        return False

    def _send_procs_to_workers(self, updatehash=False, graph=None):
        """
        Sends jobs to workers
        """
        while not np.all(self.proc_done):
            num_jobs = len(self.pending_tasks)
            if np.isinf(self.max_jobs):
                slots = None
            else:
                slots = max(0, self.max_jobs - num_jobs)
            logger.debug('Slots available: %s', slots)
            if num_jobs >= self.max_jobs or slots == 0:
                break
            jobids = np.nonzero(~self.proc_done & (self.depidx.sum(0) == 0))[1]
            if len(jobids) > 0:
                logger.info('Pending[%d] Submitting[%d] jobs Slots[%s]', num_jobs, len(jobids[:slots]), slots or 'inf')
                for jobid in jobids[:slots]:
                    if isinstance(self.procs[jobid], MapNode):
                        try:
                            num_subnodes = self.procs[jobid].num_subnodes()
                        except Exception:
                            self._clean_queue(jobid, graph)
                            self.proc_pending[jobid] = False
                            continue
                        if num_subnodes > 1:
                            submit = self._submit_mapnode(jobid)
                            if not submit:
                                continue
                    self.proc_done[jobid] = True
                    self.proc_pending[jobid] = True
                    logger.info('Submitting: %s ID: %d', self.procs[jobid], jobid)
                    if self._status_callback:
                        self._status_callback(self.procs[jobid], 'start')
                    if not self._local_hash_check(jobid, graph):
                        if self.procs[jobid].run_without_submitting:
                            logger.debug('Running node %s on master thread', self.procs[jobid])
                            try:
                                self.procs[jobid].run()
                            except Exception:
                                self._clean_queue(jobid, graph)
                            self._task_finished_cb(jobid)
                            self._remove_node_dirs()
                        else:
                            tid = self._submit_job(deepcopy(self.procs[jobid]), updatehash=updatehash)
                            if tid is None:
                                self.proc_done[jobid] = False
                                self.proc_pending[jobid] = False
                            else:
                                self.pending_tasks.insert(0, (tid, jobid))
                    logger.info('Finished submitting: %s ID: %d', self.procs[jobid], jobid)
            else:
                break

    def _local_hash_check(self, jobid, graph):
        if not str2bool(self.procs[jobid].config['execution']['local_hash_check']):
            return False
        try:
            cached, updated = self.procs[jobid].is_cached()
        except Exception:
            logger.warning('Error while checking node hash, forcing re-run. Although this error may not prevent the workflow from running, it could indicate a major problem. Please report a new issue at https://github.com/nipy/nipype/issues adding the following information:\n\n\tNode: %s\n\tInterface: %s.%s\n\tTraceback:\n%s', self.procs[jobid], self.procs[jobid].interface.__module__, self.procs[jobid].interface.__class__.__name__, '\n'.join(format_exception(*sys.exc_info())))
            return False
        logger.debug('Checking hash "%s" locally: cached=%s, updated=%s.', self.procs[jobid], cached, updated)
        overwrite = self.procs[jobid].overwrite
        always_run = self.procs[jobid].interface.always_run
        if cached and updated and (overwrite is False or (overwrite is None and (not always_run))):
            logger.debug('Skipping cached node %s with ID %s.', self.procs[jobid], jobid)
            try:
                self._task_finished_cb(jobid, cached=True)
                self._remove_node_dirs()
            except Exception:
                logger.debug('Error skipping cached node %s (%s).\n\n%s', self.procs[jobid], jobid, '\n'.join(format_exception(*sys.exc_info())))
                self._clean_queue(jobid, graph)
                self.proc_pending[jobid] = False
            return True
        return False

    def _task_finished_cb(self, jobid, cached=False):
        """Extract outputs and assign to inputs of dependent tasks

        This is called when a job is completed.
        """
        logger.info('[Job %d] %s (%s).', jobid, 'Cached' if cached else 'Completed', self.procs[jobid])
        if self._status_callback:
            self._status_callback(self.procs[jobid], 'end')
        self.proc_pending[jobid] = False
        rowview = self.depidx.getrowview(jobid)
        rowview[rowview.nonzero()] = 0
        if jobid not in self.mapnodesubids:
            self.refidx[self.refidx[:, jobid].nonzero()[0], jobid] = 0

    def _generate_dependency_list(self, graph):
        """Generates a dependency list for a list of graphs."""
        self.procs, _ = topological_sort(graph)
        self.depidx = _graph_to_lil_matrix(graph, nodelist=self.procs)
        self.refidx = self.depidx.astype(int)
        self.proc_done = np.zeros(len(self.procs), dtype=bool)
        self.proc_pending = np.zeros(len(self.procs), dtype=bool)

    def _remove_node_deps(self, jobid, crashfile, graph):
        import networkx as nx
        try:
            dfs_preorder = nx.dfs_preorder
        except AttributeError:
            dfs_preorder = nx.dfs_preorder_nodes
        subnodes = [s for s in dfs_preorder(graph, self.procs[jobid])]
        for node in subnodes:
            idx = self.procs.index(node)
            self.proc_done[idx] = True
            self.proc_pending[idx] = False
        return dict(node=self.procs[jobid], dependents=subnodes, crashfile=crashfile)

    def _remove_node_dirs(self):
        """Removes directories whose outputs have already been used up"""
        if str2bool(self._config['execution']['remove_node_directories']):
            indices = np.nonzero((self.refidx.sum(axis=1) == 0).__array__())[0]
            for idx in indices:
                if idx in self.mapnodesubids:
                    continue
                if self.proc_done[idx] and (not self.proc_pending[idx]):
                    self.refidx[idx, idx] = -1
                    outdir = self.procs[idx].output_dir()
                    logger.info('[node dependencies finished] removing node: %s from directory %s' % (self.procs[idx]._id, outdir))
                    shutil.rmtree(outdir)