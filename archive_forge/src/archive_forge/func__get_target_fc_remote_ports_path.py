from __future__ import annotations
import glob
import os
from typing import Iterable
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.initiator import linuxscsi
def _get_target_fc_remote_ports_path(self, path, wwpn, lun):
    """Scan target in the fc_remote_ports path

        Scan for target in the following path:
        * /sys/class/fc_remote_ports/rport-<host>*

        If the path exist, we fetch the target value from the
        scsi_target_id file.
        Example: /sys/class/fc_remote_ports/rport-6:0-1/scsi_target_id

        :returns: List with [c, t, l] if the target path exists else
        empty list
        """
    cmd = 'grep -Gil "%(wwpns)s" %(path)s*/port_name' % {'wwpns': wwpn, 'path': path}
    out, _err = self._execute(cmd, shell=True)
    target_path = os.path.dirname(out) + '/scsi_target_id'
    if target_path.startswith(path):
        try:
            scsi_target = '-1'
            with open(target_path) as scsi_target_file:
                lines = scsi_target_file.read()
                scsi_target = lines.split('\n')[0]
        except OSError:
            pass
        if scsi_target != '-1':
            channel = target_path.split(':')[1].split('-')[0]
            return [channel, scsi_target, lun]
    return []