from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def ParseBasicLevelConditions(api_version):
    """Wrapper around ParseCustomLevel to accept api version."""

    def VersionedParseBasicLevelConditions(path):
        """Parse a YAML representation of basic level conditions.

    Args:
      path: str, path to file containing basic level conditions

    Returns:
      list of Condition objects.

    Raises:
      ParseError: if the file could not be read into the proper object
    """
        data = yaml.load_path(path)
        if not data:
            raise ParseError(path, 'File is empty')
        messages = util.GetMessages(version=api_version)
        message_class = messages.Condition
        try:
            conditions = [encoding.DictToMessage(c, message_class) for c in data]
        except Exception as err:
            raise InvalidFormatError(path, six.text_type(err), message_class)
        _ValidateAllBasicConditionFieldsRecognized(path, conditions)
        return conditions
    return VersionedParseBasicLevelConditions