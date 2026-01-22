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
def ReportError(is_crash):
    """Report the anonymous crash information to the Error Reporting service.

  This will report the actively handled exception.
  Args:
    is_crash: bool, True if this is a crash, False if it is a user error.
  """
    if not properties.IsDefaultUniverse() or properties.VALUES.core.disable_usage_reporting.GetBool():
        return
    stacktrace = traceback.format_exc()
    stacktrace = error_reporting_util.RemovePrivateInformationFromTraceback(stacktrace)
    command = properties.VALUES.metrics.command_name.Get()
    cid = metrics.GetCIDIfMetricsEnabled()
    client = _GetReportingClient(is_crash)
    reporter = util.ErrorReporting(client)
    try:
        method_config = client.projects_events.GetMethodConfig('Report')
        request = reporter.GenerateReportRequest(error_message=stacktrace, service=SERVICE, version=config.CLOUD_SDK_VERSION, project=CRASH_PROJECT if is_crash else ERROR_PROJECT, request_url=command, user=cid)
        http_request = client.projects_events.PrepareHttpRequest(method_config, request)
        metrics.CustomBeacon(http_request.url, http_request.http_method, http_request.body, http_request.headers)
    except apitools_exceptions.Error as e:
        log.file_only_logger.error('Unable to report crash stacktrace:\n{0}'.format(console_attr.SafeText(e)))