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
def ListDebuggees(self, include_inactive=False, include_stale=False):
    """Lists all debug targets registered with the debug service.

    Args:
      include_inactive: If true, also include debuggees that are not currently
        running.
      include_stale: If false, filter out any debuggees that refer to
        stale minor versions. A debugge represents a stale minor version if it
        meets the following criteria:
            1. It has a minorversion label.
            2. All other debuggees with the same name (i.e., all debuggees with
               the same module and version, in the case of app engine) have a
               minorversion label.
            3. The minorversion value for the debuggee is less than the
               minorversion value for at least one other debuggee with the same
               name.
    Returns:
      [Debuggee] A list of debuggees.
    """
    request = self._debug_messages.ClouddebuggerDebuggerDebuggeesListRequest(project=self._project, includeInactive=include_inactive, clientVersion=self.CLIENT_VERSION)
    try:
        response = self._debug_client.debugger_debuggees.List(request)
    except apitools_exceptions.HttpError as error:
        raise errors.UnknownHttpError(error)
    result = [Debuggee(debuggee) for debuggee in response.debuggees]
    if not include_stale:
        return _FilterStaleMinorVersions(result)
    return result