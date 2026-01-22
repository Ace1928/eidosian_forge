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
def AddTargetInfo(self, message):
    """Converts a message into an object with added debuggee information.

    Args:
      message: A message returned from a debug API call.
    Returns:
      An object including the fields of the original object plus the following
      fields: project, target_uniquifier, and target_id.
    """
    result = _MessageDict(message, hidden_fields={'project': self.project, 'target_uniquifier': self.target_uniquifier, 'target_id': self.target_id, 'service': self.service, 'version': self.version})
    if message.action == self._debug_messages.Breakpoint.ActionValueValuesEnum.LOG and (not message.logLevel):
        result['logLevel'] = self._debug_messages.Breakpoint.LogLevelValueValuesEnum.INFO
    if message.isFinalState is None:
        result['isFinalState'] = False
    if message.location:
        result['location'] = _FormatLocation(message.location)
    if message.logMessageFormat:
        result['logMessageFormat'] = MergeLogExpressions(message.logMessageFormat, message.expressions)
        result.HideExistingField('expressions')
    if not message.status or not message.status.isError:
        if message.action == self.BreakpointAction(self.LOGPOINT_TYPE):
            if self.minorversion:
                result['logQuery'] = LogQueryV2String(result)
                result['logViewUrl'] = LogViewUrl(result)
        else:
            result['consoleViewUrl'] = DebugViewUrl(result)
    return result