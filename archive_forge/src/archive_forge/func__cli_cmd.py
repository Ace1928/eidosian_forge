import os
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick import utils
def _cli_cmd(self, method, volume_name):
    LOG.debug('Enter into _cli_cmd.')
    if not self.iscliexist:
        msg = _("SDS command line doesn't exist, can't execute SDS command.")
        raise exception.BrickException(message=msg)
    if not method or volume_name is None:
        return
    cmd = [self.cli_path, '-c', method, '-v', volume_name]
    out, clilog = self._execute(*cmd, run_as_root=False, root_helper=self._root_helper)
    analyse_result = self._analyze_output(out)
    LOG.debug('%(method)s volume returns %(analyse_result)s.', {'method': method, 'analyse_result': analyse_result})
    if clilog:
        LOG.error('SDS CLI output some log: %s.', clilog)
    return analyse_result