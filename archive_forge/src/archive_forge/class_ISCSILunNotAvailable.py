import sys
from os_win._i18n import _
class ISCSILunNotAvailable(ISCSITargetException):
    msg_fmt = _('Could not find lun %(target_lun)s for iSCSI target %(target_iqn)s.')