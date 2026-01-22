import sys
from os_win._i18n import _
class DiskUpdateError(OSWinException):
    msg_fmt = _('Failed to update the specified disk.')