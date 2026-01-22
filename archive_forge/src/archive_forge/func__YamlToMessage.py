from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from apitools.base.protorpclite import messages as proto_messages
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def _YamlToMessage(structured_data, msg_type, msg_friendly_name, skip_camel_case=None, path=None):
    """Load a proto message from a file containing JSON or YAML text.

  Args:
    structured_data: Dict containing the decoded YAML data.
    msg_type: The protobuf message type to create.
    msg_friendly_name: A readable name for the message type, for use in error
      messages.
    skip_camel_case: Contains proto field names or map keys whose values should
      not have camel case applied.
    path: str or None. Optional path to be used in error messages.

  Raises:
    ParseProtoException: If there was a problem interpreting the file as the
    given message type.

  Returns:
    Proto message, The message that got decoded.
  """
    structured_data = SnakeToCamel(structured_data, skip_camel_case)
    try:
        msg = _UnpackCheckUnused(structured_data, msg_type)
    except Exception as e:
        raise cloudbuild_exceptions.ParseProtoException(path, msg_friendly_name, '%s' % e)
    return msg