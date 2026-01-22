import sys
from os_win._i18n import _
class JobTerminateFailed(HyperVException):
    msg_fmt = _('Could not terminate the requested job(s).')