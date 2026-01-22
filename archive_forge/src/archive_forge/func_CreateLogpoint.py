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
def CreateLogpoint(self, location, log_format_string, log_level=None, condition=None, user_email=None, labels=None):
    """Creates a logpoint in the debuggee.

    Args:
      location: The breakpoint source location, which will be interpreted by
        the debug agents on the machines running the Debuggee. Usually of the
        form file:line-number
      log_format_string: The message to log, optionally containin {expression}-
        style formatting.
      log_level: String (case-insensitive), one of 'info', 'warning', or
        'error', indicating the log level that should be used for logging.
      condition: An optional conditional expression in the target's programming
        language. The snapshot will be taken when the expression is true.
      user_email: The email of the user who created the snapshot.
      labels: A dictionary containing key-value pairs which will be stored
        with the snapshot definition and reported when the snapshot is queried.
    Returns:
      The created Breakpoint message.
    Raises:
      InvalidLocationException: if location is empty or malformed.
      InvalidLogFormatException: if log_format is empty or malformed.
    """
    if not location:
        raise errors.InvalidLocationException('The location must not be empty.')
    if not log_format_string:
        raise errors.InvalidLogFormatException('The log format string must not be empty.')
    labels_value = None
    if labels:
        labels_value = self._debug_messages.Breakpoint.LabelsValue(additionalProperties=[self._debug_messages.Breakpoint.LabelsValue.AdditionalProperty(key=key, value=value) for key, value in six.iteritems(labels)])
    location = self._LocationFromString(location)
    if log_level:
        log_level = self._debug_messages.Breakpoint.LogLevelValueValuesEnum(log_level.upper())
    log_message_format, expressions = SplitLogExpressions(log_format_string)
    request = self._debug_messages.ClouddebuggerDebuggerDebuggeesBreakpointsSetRequest(debuggeeId=self.target_id, breakpoint=self._debug_messages.Breakpoint(location=location, condition=condition, logLevel=log_level, logMessageFormat=log_message_format, expressions=expressions, labels=labels_value, userEmail=user_email, action=self._debug_messages.Breakpoint.ActionValueValuesEnum.LOG), clientVersion=self.CLIENT_VERSION)
    try:
        response = self._debug_client.debugger_debuggees_breakpoints.Set(request)
    except apitools_exceptions.HttpError as error:
        raise errors.UnknownHttpError(error)
    return self.AddTargetInfo(response.breakpoint)