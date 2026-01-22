import logging
from oslo_vmware._i18n import _
class FileNotFoundException(VimException):
    msg_fmt = _('File not found.')
    code = 404