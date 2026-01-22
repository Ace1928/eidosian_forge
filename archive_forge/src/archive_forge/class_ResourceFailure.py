import sys
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common.i18n import _
class ResourceFailure(HeatExceptionWithPath):

    def __init__(self, exception_or_error, resource, action=None):
        self.resource = resource
        self.action = action
        if action is None and resource is not None:
            self.action = resource.action
        path = []
        res_path = []
        if resource is not None:
            res_path = [resource.stack.t.get_section_name('resources'), resource.name]
        if isinstance(exception_or_error, Exception):
            if isinstance(exception_or_error, ResourceFailure):
                self.exc = exception_or_error.exc
                error = exception_or_error.error
                message = exception_or_error.error_message
                path = exception_or_error.path
            else:
                self.exc = exception_or_error
                error = str(type(self.exc).__name__)
                message = str(self.exc)
                path = res_path
        else:
            self.exc = None
            res_failed = 'Resource %s failed: ' % self.action.upper()
            if res_failed in exception_or_error:
                error, message, new_path = self._from_status_reason(exception_or_error)
                path = res_path + new_path
            else:
                path = res_path
                error = None
                message = exception_or_error
        super(ResourceFailure, self).__init__(error=error, path=path, message=message)

    def _from_status_reason(self, status_reason):
        """Split the status_reason up into parts.

        Given the following status_reason:
        "Resource DELETE failed: Exception : resources.AResource: foo"

        we are going to return:
        ("Exception", "resources.AResource", "foo")
        """
        parsed = [sp.strip() for sp in status_reason.split(':')]
        if len(parsed) >= 4:
            error = parsed[1]
            message = ': '.join(parsed[3:])
            path = parsed[2].split('.')
        else:
            error = ''
            message = status_reason
            path = []
        return (error, message, path)