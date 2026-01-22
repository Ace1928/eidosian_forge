from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.domains import operations
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def ParseMessageFromYamlFile(path, message_type, error_message):
    """Parse a Yaml file.

  Args:
    path: Yaml file path. If path is None returns None.
    message_type: Message type to parse YAML into.
    error_message: Error message to print in case of parsing error.

  Returns:
    parsed message of type message_type.
  """
    if path is None:
        return None
    raw_message = yaml.load_path(path)
    try:
        parsed_message = encoding.PyValueToMessage(message_type, raw_message)
    except Exception as e:
        raise exceptions.Error('{}: {}'.format(error_message, e))
    unknown_fields = []
    for message in encoding.UnrecognizedFieldIter(parsed_message):
        outer_message = ''.join([edge.field + '.' for edge in message[0]])
        unknown_fields += [outer_message + field for field in message[1]]
    unknown_fields.sort()
    if unknown_fields:
        raise exceptions.Error("{}.\nProblematic fields: '{}'".format(error_message, ', '.join(unknown_fields)))
    return parsed_message