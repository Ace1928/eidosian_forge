from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
class AbortException(StandardError):
    """Exception raised when a user aborts a command that needs to do cleanup."""

    def __init__(self, reason):
        StandardError.__init__(self)
        self.reason = reason

    def __repr__(self):
        return 'AbortException: %s' % self.reason

    def __str__(self):
        return 'AbortException: %s' % self.reason