import collections
import time
from os_win import exceptions as os_win_exc
from os_win import utilsfactory
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.windows import base as win_conn_base
from os_brick import utils
def _get_dev_nums_by_scsi_id(self, local_port_wwn, remote_port_wwn, fcp_lun):
    LOG.debug('Fetching SCSI Unique ID for FCP lun %(fcp_lun)s. Port WWN: %(local_port_wwn)s. Remote port WWN: %(remote_port_wwn)s.', dict(fcp_lun=fcp_lun, local_port_wwn=local_port_wwn, remote_port_wwn=remote_port_wwn))
    local_hba_wwn = self._get_fc_hba_wwn_for_port(local_port_wwn)
    identifiers = self._fc_utils.get_scsi_device_identifiers(local_hba_wwn, local_port_wwn, remote_port_wwn, fcp_lun)
    if identifiers:
        identifier = identifiers[0]
        dev_nums = self._diskutils.get_disk_numbers_by_unique_id(unique_id=identifier['id'], unique_id_format=identifier['type'])
        return dev_nums
    return []