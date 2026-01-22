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
def WaitForBreakpoint(self, breakpoint_id, timeout=None, retry_ms=500, completion_test=None):
    """Waits for a breakpoint to be completed.

    Args:
      breakpoint_id: A breakpoint ID.
      timeout: The number of seconds to wait for completion.
      retry_ms: Milliseconds to wait betweeen retries.
      completion_test: A function that accepts a Breakpoint message and
        returns True if the breakpoint wait is not finished. If not specified,
        defaults to a function which just checks the isFinalState flag.
    Returns:
      The Breakpoint message, or None if the breakpoint did not complete before
      the timeout,
    """
    if not completion_test:
        completion_test = lambda r: r.breakpoint.isFinalState
    retry_if = lambda r, _: not completion_test(r)
    retryer = retry.Retryer(max_wait_ms=1000 * timeout if timeout is not None else None, wait_ceiling_ms=1000)
    request = self._debug_messages.ClouddebuggerDebuggerDebuggeesBreakpointsGetRequest(breakpointId=breakpoint_id, debuggeeId=self.target_id, clientVersion=self.CLIENT_VERSION)
    try:
        result = retryer.RetryOnResult(self._CallGet, [request], should_retry_if=retry_if, sleep_ms=retry_ms)
    except retry.RetryException:
        return None
    except apitools_exceptions.HttpError as error:
        raise errors.UnknownHttpError(error)
    if not completion_test(result):
        return None
    return self.AddTargetInfo(result.breakpoint)