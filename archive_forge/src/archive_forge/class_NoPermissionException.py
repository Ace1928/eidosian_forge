import logging
from oslo_vmware._i18n import _
class NoPermissionException(VimException):
    msg_fmt = _('No Permission.')
    code = 403