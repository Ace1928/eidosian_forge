from __future__ import annotations
import glob
import os
from typing import Iterable
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.initiator import linuxscsi
def _get_target_fc_transport_path(self, path, wwpn, lun):
    """Scan target in the fc_transport path

        Scan for target in the following path:
        * /sys/class/fc_transport/target<host>*

        :returns: List with [c, t, l] if the target path exists else
        empty list
        """
    cmd = 'grep -Gil "%(wwpns)s" %(path)s*/port_name' % {'wwpns': wwpn, 'path': path}
    out, _err = self._execute(cmd, shell=True)
    out_path = out.split('\n')[0]
    if out_path.startswith(path):
        return out_path.split('/')[4].split(':')[1:] + [lun]
    return []