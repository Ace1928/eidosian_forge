from typing import Optional
from twisted.python.deprecate import getWarningMethod
from twisted.python.failure import Failure
from twisted.python.log import err
from twisted.python.reflect import qual
def _callProcessExited(self, reason):
    default = object()
    processExited = getattr(self.proto, 'processExited', default)
    if processExited is default:
        getWarningMethod()(_missingProcessExited % (qual(self.proto.__class__),), DeprecationWarning, stacklevel=0)
    else:
        try:
            processExited(Failure(reason))
        except BaseException:
            err(None, 'unexpected error in processExited')