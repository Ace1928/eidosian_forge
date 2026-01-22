from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import hostutils
from os_win.utils import pathutils
from os_win.utils import win32utils
def add_disk_to_target(self, wtd_name, target_name):
    """Adds the disk to the target."""
    try:
        wt_disk = self._get_wt_disk(wtd_name)
        wt_host = self._get_wt_host(target_name)
        wt_host.AddWTDisk(wt_disk.WTD)
    except exceptions.x_wmi as wmi_exc:
        err_msg = _('Could not add WTD Disk %(wtd_name)s to iSCSI target %(target_name)s.')
        raise exceptions.ISCSITargetWMIException(err_msg % dict(wtd_name=wtd_name, target_name=target_name), wmi_exc=wmi_exc)