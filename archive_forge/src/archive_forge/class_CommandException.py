from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
class CommandException(StandardError):
    """Exception raised when a problem is encountered running a gsutil command.

  This exception should be used to signal user errors or system failures
  (like timeouts), not bugs (like an incorrect param value). For the
  latter you should raise Exception so we can see where/how it happened
  via gsutil -D (which will include a stack trace for raised Exceptions).
  """

    def __init__(self, reason, informational=False):
        """Instantiate a CommandException.

    Args:
      reason: Text describing the problem.
      informational: Indicates reason should be printed as FYI, not a failure.
    """
        StandardError.__init__(self)
        self.reason = reason
        self.informational = informational

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'CommandException: %s' % self.reason