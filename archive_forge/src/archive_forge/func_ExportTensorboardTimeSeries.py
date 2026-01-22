from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def ExportTensorboardTimeSeries(self, request, global_params=None):
    """Exports a TensorboardTimeSeries' data. Data is returned in paginated responses.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesExportTensorboardTimeSeriesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ExportTensorboardTimeSeriesDataResponse) The response message.
      """
    config = self.GetMethodConfig('ExportTensorboardTimeSeries')
    return self._RunMethod(config, request, global_params=global_params)