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
def DefaultDebuggee(self):
    """Find the default debuggee.

    Returns:
      The default debug target, which is either the only target available
      or the latest minor version of the application, if all targets have the
      same module and version.
    Raises:
      errors.NoDebuggeeError if no debuggee was found.
      errors.MultipleDebuggeesError if there is not a unique default.
    """
    debuggees = self.ListDebuggees()
    if len(debuggees) == 1:
        return debuggees[0]
    if not debuggees:
        raise errors.NoDebuggeeError()
    raise errors.MultipleDebuggeesError(None, debuggees)