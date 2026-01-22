import os
import sys
from ...interfaces.base import CommandLine
from .base import GraphPluginBase, logger
class SGEGraphPlugin(GraphPluginBase):
    """Execute using SGE

    The plugin_args input to run can be used to control the SGE execution.
    Currently supported options are:

    - template : template to use for batch job submission
    - qsub_args : arguments to be prepended to the job execution script in the
                  qsub call

    """
    _template = '\n#!/bin/bash\n#$ -V\n#$ -S /bin/bash\n'

    def __init__(self, **kwargs):
        self._qsub_args = ''
        self._dont_resubmit_completed_jobs = False
        if 'plugin_args' in kwargs and kwargs['plugin_args']:
            plugin_args = kwargs['plugin_args']
            if 'template' in plugin_args:
                self._template = plugin_args['template']
                if os.path.isfile(self._template):
                    self._template = open(self._template).read()
            if 'qsub_args' in plugin_args:
                self._qsub_args = plugin_args['qsub_args']
            if 'dont_resubmit_completed_jobs' in plugin_args:
                self._dont_resubmit_completed_jobs = plugin_args['dont_resubmit_completed_jobs']
        super(SGEGraphPlugin, self).__init__(**kwargs)

    def _submit_graph(self, pyfiles, dependencies, nodes):

        def make_job_name(jobnumber, nodeslist):
            """
            - jobnumber: The index number of the job to create
            - nodeslist: The name of the node being processed
            - return: A string representing this job to be displayed by SGE
            """
            job_name = 'j{0}_{1}'.format(jobnumber, nodeslist[jobnumber]._id)
            job_name = job_name.replace('-', '_').replace('.', '_').replace(':', '_')
            return job_name
        batch_dir, _ = os.path.split(pyfiles[0])
        submitjobsfile = os.path.join(batch_dir, 'submit_jobs.sh')
        cache_doneness_per_node = dict()
        if self._dont_resubmit_completed_jobs:
            for idx, pyscript in enumerate(pyfiles):
                node = nodes[idx]
                node_status_done = node_completed_status(node)
                if node_status_done and idx in dependencies:
                    for child_idx in dependencies[idx]:
                        if child_idx in cache_doneness_per_node:
                            child_status_done = cache_doneness_per_node[child_idx]
                        else:
                            child_status_done = node_completed_status(nodes[child_idx])
                        node_status_done = node_status_done and child_status_done
                cache_doneness_per_node[idx] = node_status_done
        with open(submitjobsfile, 'wt') as fp:
            fp.writelines('#!/usr/bin/env bash\n')
            fp.writelines('# Condense format attempted\n')
            for idx, pyscript in enumerate(pyfiles):
                node = nodes[idx]
                if cache_doneness_per_node.get(idx, False):
                    continue
                else:
                    template, qsub_args = self._get_args(node, ['template', 'qsub_args'])
                    batch_dir, name = os.path.split(pyscript)
                    name = '.'.join(name.split('.')[:-1])
                    batchscript = '\n'.join((template, '%s %s' % (sys.executable, pyscript)))
                    batchscriptfile = os.path.join(batch_dir, 'batchscript_%s.sh' % name)
                    batchscriptoutfile = batchscriptfile + '.o'
                    batchscripterrfile = batchscriptfile + '.e'
                    with open(batchscriptfile, 'wt') as batchfp:
                        batchfp.writelines(batchscript)
                        batchfp.close()
                    deps = ''
                    if idx in dependencies:
                        values = ' '
                        for jobid in dependencies[idx]:
                            if not self._dont_resubmit_completed_jobs or not cache_doneness_per_node[jobid]:
                                values += '${{{0}}},'.format(make_job_name(jobid, nodes))
                        if values != ' ':
                            values = values.rstrip(',')
                            deps = '-hold_jid%s' % values
                    jobname = make_job_name(idx, nodes)
                    stderrFile = ''
                    if self._qsub_args.count('-e ') == 0:
                        stderrFile = '-e {errFile}'.format(errFile=batchscripterrfile)
                    stdoutFile = ''
                    if self._qsub_args.count('-o ') == 0:
                        stdoutFile = '-o {outFile}'.format(outFile=batchscriptoutfile)
                    full_line = "{jobNm}=$(qsub {outFileOption} {errFileOption} {extraQSubArgs} {dependantIndex} -N {jobNm} {batchscript} | awk '/^Your job/{{print $3}}')\n".format(jobNm=jobname, outFileOption=stdoutFile, errFileOption=stderrFile, extraQSubArgs=qsub_args, dependantIndex=deps, batchscript=batchscriptfile)
                    fp.writelines(full_line)
        cmd = CommandLine('bash', environ=dict(os.environ), resource_monitor=False, terminal_output='allatonce')
        cmd.inputs.args = '%s' % submitjobsfile
        cmd.run()
        logger.info('submitted all jobs to queue')