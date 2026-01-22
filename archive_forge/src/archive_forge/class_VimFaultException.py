import logging
from oslo_vmware._i18n import _
class VimFaultException(VimException):
    """Exception thrown when there are unrecognized VIM faults."""

    def __init__(self, fault_list, message, cause=None, details=None):
        super(VimFaultException, self).__init__(message, cause, details)
        if not isinstance(fault_list, list):
            raise ValueError(_('fault_list must be a list'))
        self.fault_list = fault_list

    @property
    def description(self):
        descr = VimException.description.fget(self)
        if self.fault_list:
            descr += '\nFaults: ' + str(self.fault_list)
        if self.details:
            details = '{%s}' % ', '.join(["'%s': '%s'" % (k, v) for k, v in self.details.items()])
            descr += '\nDetails: ' + details
        return descr