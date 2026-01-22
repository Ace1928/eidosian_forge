from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
class NoDebuggeeError(DebugError):
    """No debug target matched the search criteria."""

    def __init__(self, pattern=None, debuggees=None):
        if pattern:
            msg = 'No active debug target matched the pattern "{0}"\n'.format(pattern)
        else:
            msg = 'No active debug targets were found for this project.\n'
        if debuggees:
            msg += 'Use the --target option to select one of the following targets:\n    {0}\n'.format('\n    '.join([d.name for d in debuggees]))
        super(NoDebuggeeError, self).__init__(msg)