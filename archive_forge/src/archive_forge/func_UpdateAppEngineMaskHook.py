from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_encoding
def UpdateAppEngineMaskHook(unused_ref, args, req):
    """Constructs updateMask for patch requests of AppEngine targets.

  Args:
    unused_ref: A resource ref to the parsed Job resource
    args: The parsed args namespace from CLI
    req: Created Patch request for the API call.

  Returns:
    Modified request for the API call.
  """
    app_engine_fields = {'--message-body': 'appEngineHttpTarget.body', '--message-body-from-file': 'appEngineHttpTarget.body', '--relative-url': 'appEngineHttpTarget.relativeUri', '--version': 'appEngineHttpTarget.appEngineRouting.version', '--service': 'appEngineHttpTarget.appEngineRouting.service', '--clear-service': 'appEngineHttpTarget.appEngineRouting.service', '--clear-relative-url': 'appEngineHttpTarget.relativeUri', '--clear-headers': 'appEngineHttpTarget.headers', '--remove-headers': 'appEngineHttpTarget.headers', '--update-headers': 'appEngineHttpTarget.headers'}
    req.updateMask = _GenerateUpdateMask(args, app_engine_fields)
    return req