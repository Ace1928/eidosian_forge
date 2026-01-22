from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import io
import json
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_diff
from googlecloudsdk.core.util import edit
import six
def ParseAuthorizedViewFromYamlOrJsonDefinitionFile(file_path, pre_encoded):
    """Create an authorized view proto message from a YAML or JSON formatted definition file.

  Args:
    file_path: Path to the YAML or JSON definition file.
    pre_encoded: True if all binary fields in the authorized view definition are
      already Base64-encoded. We skip the step of applying Base64 encoding in
      this case.

  Returns:
    an authorized view proto message with fields filled accordingly.

  Raises:
    BadArgumentException if the file cannot be read.
    BadArgumentException if the file does not follow YAML or JSON format.
    ValueError if the YAML/JSON object cannot be parsed as a valid authorized
      view.
  """
    authorized_view_message_type = util.GetAdminMessages().AuthorizedView
    try:
        authorized_view_to_parse = yaml.load_path(file_path)
        if not pre_encoded:
            Base64EncodingYamlAuthorizedViewDefinition(authorized_view_to_parse)
        authorized_view = encoding.PyValueToMessage(authorized_view_message_type, authorized_view_to_parse)
    except (yaml.FileLoadError, yaml.YAMLParseError) as e:
        raise calliope_exceptions.BadArgumentException('--definition-file', e)
    except AttributeError as e:
        raise ValueError('File [{0}] cannot be parsed as a valid authorized view. [{1}].'.format(file_path, e))
    return authorized_view