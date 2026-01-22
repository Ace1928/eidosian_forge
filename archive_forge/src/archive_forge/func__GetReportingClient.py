from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import sys
import traceback
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.error_reporting import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib import error_reporting_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import platforms
def _GetReportingClient(is_crash=True):
    """Returns a client that uses an API key for Cloud SDK crash reports.

  Args:
     is_crash: bool, True use CRASH_REPORTING_PARAM, if False use
     ERROR_REPORTING_PARAM.

  Returns:
    An error reporting client that uses an API key for Cloud SDK crash reports.
  """
    client_class = core_apis.GetClientClass(util.API_NAME, util.API_VERSION)
    client_instance = client_class(get_credentials=False)
    if is_crash:
        client_instance.AddGlobalParam('key', CRASH_REPORTING_PARAM)
    else:
        client_instance.AddGlobalParam('key', ERROR_REPORTING_PARAM)
    return client_instance