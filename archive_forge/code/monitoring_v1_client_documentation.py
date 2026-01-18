from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v1 import monitoring_v1_messages as messages
Lists metadata for metrics.

      Args:
        request: (MonitoringProjectsLocationPrometheusApiV1SeriesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      