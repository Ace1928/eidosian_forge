from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.compute import ssh_troubleshooter_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console.console_io import OperationCancelledError
def _GetCpuUtilization(self):
    """Get CPU utilization from Cloud Monitoring API."""
    for req_field, mapped_param in _CUSTOM_JSON_FIELD_MAPPINGS.items():
        encoding.AddCustomJsonFieldMapping(self.monitoring_message.MonitoringProjectsTimeSeriesListRequest, req_field, mapped_param)
    request = self._CreateTimeSeriesListRequest(CPU_METRICS)
    response = self.monitoring_client.projects_timeSeries.List(request=request)
    if response.timeSeries:
        points = response.timeSeries[0].points
        return sum((point.value.doubleValue for point in points)) / len(points)
    return 0.0