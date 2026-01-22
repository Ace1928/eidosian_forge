import logging
from oslo_vmware._i18n import _
class VimException(VMwareDriverException):
    """The base exception class for all VIM related exceptions."""

    def __init__(self, message=None, cause=None, details=None, **kwargs):
        super(VimException, self).__init__(message, details, **kwargs)
        self.cause = cause