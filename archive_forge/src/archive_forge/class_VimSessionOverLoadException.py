import logging
from oslo_vmware._i18n import _
class VimSessionOverLoadException(VMwareDriverException):
    """Thrown when there is an API call overload at the VMware server."""

    def __init__(self, message, cause=None):
        super(VimSessionOverLoadException, self).__init__(message)
        self.cause = cause