from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1 import healthcare_v1_messages as messages
def GetSeriesMetrics(self, request, global_params=None):
    """GetSeriesMetrics returns metrics for a series.

      Args:
        request: (HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesSeriesGetSeriesMetricsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SeriesMetrics) The response message.
      """
    config = self.GetMethodConfig('GetSeriesMetrics')
    return self._RunMethod(config, request, global_params=global_params)