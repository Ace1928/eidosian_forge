from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
def FindDebuggee(self, pattern=None):
    """Find the unique debuggee matching the given pattern.

    Args:
      pattern: A string containing a debuggee ID or a regular expression that
        matches a single debuggee's name or description. If it matches any
        debuggee name, the description will not be inspected.
    Returns:
      The matching Debuggee.
    Raises:
      errors.MultipleDebuggeesError if the pattern matches multiple debuggees.
      errors.NoDebuggeeError if the pattern matches no debuggees.
    """
    if not pattern:
        debuggee = self.DefaultDebuggee()
        log.status.write('Debug target not specified. Using default target: {0}\n'.format(debuggee.name))
        return debuggee
    try:
        all_debuggees = self.ListDebuggees()
        return self._FilterDebuggeeList(all_debuggees, pattern)
    except errors.NoDebuggeeError:
        pass
    all_debuggees = self.ListDebuggees(include_inactive=True, include_stale=True)
    return self._FilterDebuggeeList(all_debuggees, pattern)