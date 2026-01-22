from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def UpsertDatapoints(self, request, global_params=None):
    """Add/update Datapoints into an Index.

      Args:
        request: (AiplatformProjectsLocationsIndexesUpsertDatapointsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1UpsertDatapointsResponse) The response message.
      """
    config = self.GetMethodConfig('UpsertDatapoints')
    return self._RunMethod(config, request, global_params=global_params)