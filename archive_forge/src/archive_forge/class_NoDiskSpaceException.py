import logging
from oslo_vmware._i18n import _
class NoDiskSpaceException(VimException):
    msg_fmt = _('Insufficient disk space.')