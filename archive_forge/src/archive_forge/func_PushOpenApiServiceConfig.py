from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def PushOpenApiServiceConfig(service_name, spec_file_contents, spec_file_path, is_async, validate_only=False):
    """Pushes a given Open API service configuration.

  Args:
    service_name: name of the service
    spec_file_contents: the contents of the Open API spec file.
    spec_file_path: the path of the Open API spec file.
    is_async: whether to wait for aync operations or not.
    validate_only: whether to perform a validate-only run of the operation
                   or not.

  Returns:
    Full response from the SubmitConfigSource request.
  """
    messages = GetMessagesModule()
    config_file = messages.ConfigFile(fileContents=spec_file_contents, filePath=spec_file_path, fileType=messages.ConfigFile.FileTypeValueValuesEnum.OPEN_API_YAML)
    return PushMultipleServiceConfigFiles(service_name, [config_file], is_async, validate_only=validate_only)