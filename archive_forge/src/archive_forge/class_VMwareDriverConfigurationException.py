import logging
from oslo_vmware._i18n import _
class VMwareDriverConfigurationException(VMwareDriverException):
    """Base class for all configuration exceptions.
    """
    msg_fmt = _('VMware Driver configuration fault.')

    def __init__(self, message=None, details=None, **kwargs):
        super(VMwareDriverConfigurationException, self).__init__(message, details, **kwargs)
        _print_deprecation_warning(self.__class__)