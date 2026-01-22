from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import textwrap
from googlecloudsdk.command_lib.util import check_browser
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def GetCredentialsConfigFromFile(filename):
    """Returns the JSON content of a credentials config file.

  This function is useful when the content of a file need to be inspected first
  before determining how to handle it (how to initialize the underlying
  credentials). Only UTF-8 JSON files are supported.

  Args:
    filename (str): The filepath to the ADC file representing credentials.

  Returns:
    Optional(Mapping): The JSON content.

  Raises:
    BadCredentialFileException: If JSON parsing of the file fails.
  """
    try:
        content = yaml.load_path(filename)
    except UnicodeDecodeError as e:
        raise BadCredentialFileException('File {0} is not utf-8 encoded: {1}'.format(filename, e))
    except yaml.YAMLParseError as e:
        raise BadCredentialFileException('Could not read json file {0}: {1}'.format(filename, e))
    if not isinstance(content, dict):
        raise BadCredentialFileException('Could not read json file {0}'.format(filename))
    return content