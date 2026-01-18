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
Reads a YAML or JSON object of type message from local path.

  Args:
    path: A local path to an object specification in YAML or JSON format.
    message: The message type to be parsed from the file.

  Returns:
    Object of type message, if successful.
  Raises:
    files.Error, exceptions.ResourceManagerInputFileError
  