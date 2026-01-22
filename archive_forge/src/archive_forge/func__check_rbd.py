import ctypes
import errno
import json
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_service import loopingcall
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base_rbd
from os_brick.initiator.windows import base as win_conn_base
from os_brick import utils
def _check_rbd(self):
    cmd = ['where.exe', 'rbd']
    try:
        self._execute(*cmd)
        return True
    except processutils.ProcessExecutionError:
        LOG.warning('rbd.exe is not available.')
    return False