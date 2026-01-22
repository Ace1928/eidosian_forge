from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import acm_printer
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def ParseAccessContextManagerMessages(path, message_class):
    """Parse a YAML representation of a list of messages.

  Args:
    path: str, path to file containing Ingress/Egress Policies
    message_class: obj, message type to parse the contents of the yaml file to

  Returns:
    list of message objects.

  Raises:
    ParseError: if the file could not be read into the proper object
  """
    data = yaml.load_path(path)
    if not data:
        raise ParseError(path, 'File is empty')
    try:
        messages = [encoding.DictToMessage(c, message_class) for c in data]
    except Exception as err:
        raise InvalidMessageParseError(path, six.text_type(err), message_class)
    _ValidateAllFieldsRecognized(path, messages)
    return messages