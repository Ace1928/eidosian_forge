import os
import stat
from time import sleep
import subprocess
import simplejson as json
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
def _submit_batchtask(self, scriptfile, node):
    cmd = CommandLine('oarsub', environ=dict(os.environ), resource_monitor=False, terminal_output='allatonce')
    path = os.path.dirname(scriptfile)
    oarsubargs = ''
    if self._oarsub_args:
        oarsubargs = self._oarsub_args
    if 'oarsub_args' in node.plugin_args:
        if 'overwrite' in node.plugin_args and node.plugin_args['overwrite']:
            oarsubargs = node.plugin_args['oarsub_args']
        else:
            oarsubargs += ' ' + node.plugin_args['oarsub_args']
    if node._hierarchy:
        jobname = '.'.join((dict(os.environ)['LOGNAME'], node._hierarchy, node._id))
    else:
        jobname = '.'.join((dict(os.environ)['LOGNAME'], node._id))
    jobnameitems = jobname.split('.')
    jobnameitems.reverse()
    jobname = '.'.join(jobnameitems)
    jobname = jobname[0:self._max_jobname_len]
    if '-O' not in oarsubargs:
        oarsubargs = '%s -O %s' % (oarsubargs, os.path.join(path, jobname + '.stdout'))
    if '-E' not in oarsubargs:
        oarsubargs = '%s -E %s' % (oarsubargs, os.path.join(path, jobname + '.stderr'))
    if '-J' not in oarsubargs:
        oarsubargs = '%s -J' % oarsubargs
    os.chmod(scriptfile, stat.S_IEXEC | stat.S_IREAD | stat.S_IWRITE)
    cmd.inputs.args = '%s -n %s -S %s' % (oarsubargs, jobname, scriptfile)
    oldlevel = iflogger.level
    iflogger.setLevel(logging.getLevelName('CRITICAL'))
    tries = 0
    while True:
        try:
            result = cmd.run()
        except Exception as e:
            if tries < self._max_tries:
                tries += 1
                sleep(self._retry_timeout)
            else:
                iflogger.setLevel(oldlevel)
                raise RuntimeError('\n'.join(('Could not submit OAR task for node %s' % node._id, str(e))))
        else:
            break
    iflogger.setLevel(oldlevel)
    o = ''
    add = False
    for line in result.runtime.stdout.splitlines():
        if line.strip().startswith('{'):
            add = True
        if add:
            o += line + '\n'
        if line.strip().startswith('}'):
            break
    taskid = json.loads(o)['job_id']
    self._pending[taskid] = node.output_dir()
    logger.debug('submitted OAR task: %s for node %s' % (taskid, node._id))
    return taskid