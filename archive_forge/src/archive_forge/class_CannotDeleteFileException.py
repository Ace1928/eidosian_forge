import logging
from oslo_vmware._i18n import _
class CannotDeleteFileException(VimException):
    msg_fmt = _('Cannot delete file.')
    code = 403