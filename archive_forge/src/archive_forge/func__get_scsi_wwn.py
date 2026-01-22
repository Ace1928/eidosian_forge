import warnings
from os_win import utilsfactory
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator import initiator_connector
from os_brick import utils
def _get_scsi_wwn(self, device_number):
    disk_uid, uid_type = self._diskutils.get_disk_uid_and_uid_type(device_number)
    scsi_wwn = '%s%s' % (uid_type, disk_uid)
    return scsi_wwn