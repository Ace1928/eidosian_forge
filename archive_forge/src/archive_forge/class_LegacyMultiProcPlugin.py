import os
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, pool
from traceback import format_exception
import sys
from logging import INFO
import gc
from copy import deepcopy
import numpy as np
from ... import logging
from ...utils.profiler import get_system_total_memory_gb
from ..engine import MapNode
from .base import DistributedPluginBase
class LegacyMultiProcPlugin(DistributedPluginBase):
    """
    Execute workflow with multiprocessing, not sending more jobs at once
    than the system can support.

    The plugin_args input to run can be used to control the multiprocessing
    execution and defining the maximum amount of memory and threads that
    should be used. When those parameters are not specified,
    the number of threads and memory of the system is used.

    System consuming nodes should be tagged::

      memory_consuming_node.mem_gb = 8
      thread_consuming_node.n_procs = 16

    The default number of threads and memory are set at node
    creation, and are 1 and 0.25GB respectively.

    Currently supported options are:

    - non_daemon : boolean flag to execute as non-daemon processes
    - n_procs: maximum number of threads to be executed in parallel
    - memory_gb: maximum memory (in GB) that can be used at once.
    - raise_insufficient: raise error if the requested resources for
        a node over the maximum `n_procs` and/or `memory_gb`
        (default is ``True``).
    - scheduler: sort jobs topologically (``'tsort'``, default value)
        or prioritize jobs by, first, memory consumption and, second,
        number of threads (``'mem_thread'`` option).
    - maxtasksperchild: number of nodes to run on each process before
        refreshing the worker (default: 10).

    """

    def __init__(self, plugin_args=None):
        super(LegacyMultiProcPlugin, self).__init__(plugin_args=plugin_args)
        self._taskresult = {}
        self._task_obj = {}
        self._taskid = 0
        self._cwd = os.getcwd()
        non_daemon = self.plugin_args.get('non_daemon', True)
        maxtasks = self.plugin_args.get('maxtasksperchild', 10)
        self.processors = self.plugin_args.get('n_procs', cpu_count())
        self.memory_gb = self.plugin_args.get('memory_gb', get_system_total_memory_gb() * 0.9)
        self.raise_insufficient = self.plugin_args.get('raise_insufficient', True)
        logger.debug('[LegacyMultiProc] Starting in "%sdaemon" mode (n_procs=%d, mem_gb=%0.2f, cwd=%s)', 'non' * int(non_daemon), self.processors, self.memory_gb, self._cwd)
        NipypePool = NonDaemonPool if non_daemon else Pool
        try:
            self.pool = NipypePool(processes=self.processors, maxtasksperchild=maxtasks, initializer=process_initializer, initargs=(self._cwd,))
        except TypeError:
            self.pool = NipypePool(processes=self.processors)
        self._stats = None

    def _async_callback(self, args):
        os.chdir(self._cwd)
        self._taskresult[args['taskid']] = args

    def _get_result(self, taskid):
        return self._taskresult.get(taskid)

    def _clear_task(self, taskid):
        del self._task_obj[taskid]

    def _submit_job(self, node, updatehash=False):
        self._taskid += 1
        if getattr(node.interface, 'terminal_output', '') == 'stream':
            node.interface.terminal_output = 'allatonce'
        self._task_obj[self._taskid] = self.pool.apply_async(run_node, (node, updatehash, self._taskid), callback=self._async_callback)
        logger.debug('[LegacyMultiProc] Submitted task %s (taskid=%d).', node.fullname, self._taskid)
        return self._taskid

    def _prerun_check(self, graph):
        """Check if any node exceeds the available resources"""
        tasks_mem_gb = []
        tasks_num_th = []
        for node in graph.nodes():
            tasks_mem_gb.append(node.mem_gb)
            tasks_num_th.append(node.n_procs)
        if np.any(np.array(tasks_mem_gb) > self.memory_gb):
            logger.warning('Some nodes exceed the total amount of memory available (%0.2fGB).', self.memory_gb)
            if self.raise_insufficient:
                raise RuntimeError('Insufficient resources available for job')
        if np.any(np.array(tasks_num_th) > self.processors):
            logger.warning('Some nodes demand for more threads than available (%d).', self.processors)
            if self.raise_insufficient:
                raise RuntimeError('Insufficient resources available for job')

    def _postrun_check(self):
        self.pool.close()

    def _check_resources(self, running_tasks):
        """
        Make sure there are resources available
        """
        free_memory_gb = self.memory_gb
        free_processors = self.processors
        for _, jobid in running_tasks:
            free_memory_gb -= min(self.procs[jobid].mem_gb, free_memory_gb)
            free_processors -= min(self.procs[jobid].n_procs, free_processors)
        return (free_memory_gb, free_processors)

    def _send_procs_to_workers(self, updatehash=False, graph=None):
        """
        Sends jobs to workers when system resources are available.
        """
        jobids = np.flatnonzero(~self.proc_done & (self.depidx.sum(axis=0) == 0).__array__())
        free_memory_gb, free_processors = self._check_resources(self.pending_tasks)
        stats = (len(self.pending_tasks), len(jobids), free_memory_gb, self.memory_gb, free_processors, self.processors)
        if self._stats != stats:
            tasks_list_msg = ''
            if logger.level <= INFO:
                running_tasks = ['  * %s' % self.procs[jobid].fullname for _, jobid in self.pending_tasks]
                if running_tasks:
                    tasks_list_msg = '\nCurrently running:\n'
                    tasks_list_msg += '\n'.join(running_tasks)
                    tasks_list_msg = indent(tasks_list_msg, ' ' * 21)
            logger.info('[LegacyMultiProc] Running %d tasks, and %d jobs ready. Free memory (GB): %0.2f/%0.2f, Free processors: %d/%d.%s', len(self.pending_tasks), len(jobids), free_memory_gb, self.memory_gb, free_processors, self.processors, tasks_list_msg)
            self._stats = stats
        if free_memory_gb < 0.01 or free_processors == 0:
            logger.debug('No resources available')
            return
        if len(jobids) + len(self.pending_tasks) == 0:
            logger.debug('No tasks are being run, and no jobs can be submitted to the queue. Potential deadlock')
            return
        jobids = self._sort_jobs(jobids, scheduler=self.plugin_args.get('scheduler'))
        gc.collect()
        for jobid in jobids:
            if isinstance(self.procs[jobid], MapNode):
                try:
                    num_subnodes = self.procs[jobid].num_subnodes()
                except Exception:
                    traceback = format_exception(*sys.exc_info())
                    self._clean_queue(jobid, graph, result={'result': None, 'traceback': traceback})
                    self.proc_pending[jobid] = False
                    continue
                if num_subnodes > 1:
                    submit = self._submit_mapnode(jobid)
                    if not submit:
                        continue
            next_job_gb = min(self.procs[jobid].mem_gb, self.memory_gb)
            next_job_th = min(self.procs[jobid].n_procs, self.processors)
            if next_job_th > free_processors or next_job_gb > free_memory_gb:
                logger.debug('Cannot allocate job %d (%0.2fGB, %d threads).', jobid, next_job_gb, next_job_th)
                continue
            free_memory_gb -= next_job_gb
            free_processors -= next_job_th
            logger.debug('Allocating %s ID=%d (%0.2fGB, %d threads). Free: %0.2fGB, %d threads.', self.procs[jobid].fullname, jobid, next_job_gb, next_job_th, free_memory_gb, free_processors)
            self.proc_done[jobid] = True
            self.proc_pending[jobid] = True
            if self._local_hash_check(jobid, graph):
                continue
            if updatehash or self.procs[jobid].run_without_submitting:
                logger.debug('Running node %s on master thread', self.procs[jobid])
                try:
                    self.procs[jobid].run(updatehash=updatehash)
                except Exception:
                    traceback = format_exception(*sys.exc_info())
                    self._clean_queue(jobid, graph, result={'result': None, 'traceback': traceback})
                self._task_finished_cb(jobid)
                self._remove_node_dirs()
                free_memory_gb += next_job_gb
                free_processors += next_job_th
                self._stats = None
                gc.collect()
                continue
            if self._status_callback:
                self._status_callback(self.procs[jobid], 'start')
            tid = self._submit_job(deepcopy(self.procs[jobid]), updatehash=updatehash)
            if tid is None:
                self.proc_done[jobid] = False
                self.proc_pending[jobid] = False
            else:
                self.pending_tasks.insert(0, (tid, jobid))
            self._stats = None

    def _sort_jobs(self, jobids, scheduler='tsort'):
        if scheduler == 'mem_thread':
            return sorted(jobids, key=lambda item: (self.procs[item].mem_gb, self.procs[item].n_procs))
        return jobids