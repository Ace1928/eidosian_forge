import sys
from os_win._i18n import _
class HyperVVMNotFoundException(NotFound, HyperVException):
    msg_fmt = _('VM not found: %(vm_name)s')