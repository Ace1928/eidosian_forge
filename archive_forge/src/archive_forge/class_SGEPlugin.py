import os
import pwd
import re
import subprocess
import time
import xml.dom.minidom
import random
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
class SGEPlugin(SGELikeBatchManagerBase):
    """Execute using SGE (OGE not tested)

    The plugin_args input to run can be used to control the SGE execution.
    Currently supported options are:

    - template : template to use for batch job submission
    - qsub_args : arguments to be prepended to the job execution script in the
                  qsub call

    """

    def __init__(self, **kwargs):
        template = '\n#$ -V\n#$ -S /bin/sh\n        '
        self._retry_timeout = 2
        self._max_tries = 2
        instant_qstat = 'qstat'
        cached_qstat = 'qstat'
        if 'plugin_args' in kwargs and kwargs['plugin_args']:
            if 'retry_timeout' in kwargs['plugin_args']:
                self._retry_timeout = kwargs['plugin_args']['retry_timeout']
            if 'max_tries' in kwargs['plugin_args']:
                self._max_tries = kwargs['plugin_args']['max_tries']
            if 'qstatProgramPath' in kwargs['plugin_args']:
                instant_qstat = kwargs['plugin_args']['qstatProgramPath']
            if 'qstatCachedProgramPath' in kwargs['plugin_args']:
                cached_qstat = kwargs['plugin_args']['qstatCachedProgramPath']
        self._refQstatSubstitute = QstatSubstitute(instant_qstat, cached_qstat)
        super(SGEPlugin, self).__init__(template, **kwargs)

    def _is_pending(self, taskid):
        return self._refQstatSubstitute.is_job_pending(int(taskid))

    def _submit_batchtask(self, scriptfile, node):
        cmd = CommandLine('qsub', environ=dict(os.environ), resource_monitor=False, terminal_output='allatonce')
        path = os.path.dirname(scriptfile)
        qsubargs = ''
        if self._qsub_args:
            qsubargs = self._qsub_args
        if 'qsub_args' in node.plugin_args:
            if 'overwrite' in node.plugin_args and node.plugin_args['overwrite']:
                qsubargs = node.plugin_args['qsub_args']
            else:
                qsubargs += ' ' + node.plugin_args['qsub_args']
        if '-o' not in qsubargs:
            qsubargs = '%s -o %s' % (qsubargs, path)
        if '-e' not in qsubargs:
            qsubargs = '%s -e %s' % (qsubargs, path)
        if node._hierarchy:
            jobname = '.'.join((dict(os.environ)['LOGNAME'], node._hierarchy, node._id))
        else:
            jobname = '.'.join((dict(os.environ)['LOGNAME'], node._id))
        jobnameitems = jobname.split('.')
        jobnameitems.reverse()
        jobname = '.'.join(jobnameitems)
        jobname = qsub_sanitize_job_name(jobname)
        cmd.inputs.args = '%s -N %s %s' % (qsubargs, jobname, scriptfile)
        oldlevel = iflogger.level
        iflogger.setLevel(logging.getLevelName('CRITICAL'))
        tries = 0
        result = list()
        while True:
            try:
                result = cmd.run()
            except Exception as e:
                if tries < self._max_tries:
                    tries += 1
                    time.sleep(self._retry_timeout)
                else:
                    iflogger.setLevel(oldlevel)
                    raise RuntimeError('\n'.join(('Could not submit sge task for node %s' % node._id, str(e))))
            else:
                break
        iflogger.setLevel(oldlevel)
        lines = [line for line in result.runtime.stdout.split('\n') if line]
        taskid = int(re.match('Your job ([0-9]*) .* has been submitted', lines[-1]).groups()[0])
        self._pending[taskid] = node.output_dir()
        self._refQstatSubstitute.add_startup_job(taskid, cmd.cmdline)
        logger.debug('submitted sge task: %d for node %s with %s' % (taskid, node._id, cmd.cmdline))
        return taskid