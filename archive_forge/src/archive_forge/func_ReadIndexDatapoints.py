from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def ReadIndexDatapoints(self, request, global_params=None):
    """Reads the datapoints/vectors of the given IDs. A maximum of 1000 datapoints can be retrieved in a batch.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsReadIndexDatapointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ReadIndexDatapointsResponse) The response message.
      """
    config = self.GetMethodConfig('ReadIndexDatapoints')
    return self._RunMethod(config, request, global_params=global_params)