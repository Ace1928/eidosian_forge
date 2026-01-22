from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def BatchRead(self, request, global_params=None):
    """Reads multiple TensorboardTimeSeries' data. The data point number limit is 1000 for scalars, 100 for tensors and blob references. If the number of data points stored is less than the limit, all data is returned. Otherwise, the number limit of data points is randomly selected from this time series and returned.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsBatchReadRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1BatchReadTensorboardTimeSeriesDataResponse) The response message.
      """
    config = self.GetMethodConfig('BatchRead')
    return self._RunMethod(config, request, global_params=global_params)