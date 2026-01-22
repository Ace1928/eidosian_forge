from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from googlecloudsdk.command_lib.util.anthos import structured_messages as sm
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def ReadStructuredOutput(msg_string, as_json=True):
    """Process a line of structured output into an OutputMessgage.

  Args:
    msg_string: string, line JSON/YAML formatted raw output text.
    as_json: boolean, if True set default string representation for parsed
      message to JSON. If False (default) use YAML.

  Returns:
    OutputMessage, parsed Message

  Raises: StructuredOutputError is msg_string can not be parsed as an
    OutputMessage.

  """
    try:
        return sm.OutputMessage.FromString(msg_string.strip(), as_json=as_json)
    except (sm.MessageParsingError, sm.InvalidMessageError) as e:
        raise StructuredOutputError('Error processing message [{msg}] as an OutputMessage: {error}'.format(msg=msg_string, error=e))