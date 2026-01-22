import sys
from os_win._i18n import _
class HyperVAuthorizationException(HyperVException):
    msg_fmt = _("The Windows account running nova-compute on this Hyper-V host doesn't have the required permissions to perform Hyper-V related operations.")