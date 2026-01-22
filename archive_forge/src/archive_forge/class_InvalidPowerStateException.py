import logging
from oslo_vmware._i18n import _
class InvalidPowerStateException(VimException):
    msg_fmt = _('Invalid power state.')
    code = 409