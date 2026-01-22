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
def _FilterDebuggeeList(self, all_debuggees, pattern):
    """Finds the debuggee which matches the given pattern.

    Args:
      all_debuggees: A list of debuggees to search.
      pattern: A string containing a debuggee ID or a regular expression that
        matches a single debuggee's name or description. If it matches any
        debuggee name, the description will not be inspected.
    Returns:
      The matching Debuggee.
    Raises:
      errors.MultipleDebuggeesError if the pattern matches multiple debuggees.
      errors.NoDebuggeeError if the pattern matches no debuggees.
    """
    if not all_debuggees:
        raise errors.NoDebuggeeError()
    latest_debuggees = _FilterStaleMinorVersions(all_debuggees)
    debuggees = [d for d in all_debuggees if d.target_id == pattern] + [d for d in latest_debuggees if pattern == d.name]
    if not debuggees:
        match_re = re.compile(pattern)
        debuggees = [d for d in latest_debuggees if match_re.search(d.name)] + [d for d in latest_debuggees if d.description and match_re.search(d.description)]
    if not debuggees:
        raise errors.NoDebuggeeError(pattern, debuggees=all_debuggees)
    debuggee_ids = set((d.target_id for d in debuggees))
    if len(debuggee_ids) > 1:
        raise errors.MultipleDebuggeesError(pattern, debuggees)
    return debuggees[0]