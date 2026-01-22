from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite.messages import DecodeError
from apitools.base.py import encoding
from googlecloudsdk.api_lib.resource_manager import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def GetFileAsMessage(path, message):
    """Reads a YAML or JSON object of type message from local path.

  Args:
    path: A local path to an object specification in YAML or JSON format.
    message: The message type to be parsed from the file.

  Returns:
    Object of type message, if successful.
  Raises:
    files.Error, exceptions.ResourceManagerInputFileError
  """
    in_text = files.ReadFileContents(path)
    if not in_text:
        raise exceptions.ResourceManagerInputFileError('Empty policy file [{0}]'.format(path))
    try:
        result = encoding.PyValueToMessage(message, yaml.load(in_text))
    except (ValueError, AttributeError, yaml.YAMLParseError):
        try:
            result = encoding.JsonToMessage(message, in_text)
        except (ValueError, DecodeError) as e:
            raise exceptions.ResourceManagerInputFileError('Policy file [{0}] is not properly formatted YAML or JSON due to [{1}]'.format(path, six.text_type(e)))
    return result