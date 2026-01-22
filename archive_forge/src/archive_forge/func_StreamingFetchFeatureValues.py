from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def StreamingFetchFeatureValues(self, request, global_params=None):
    """Bidirectional streaming RPC to fetch feature values under a FeatureView. Requests may not have a one-to-one mapping to responses and responses may be returned out-of-order to reduce latency.

      Args:
        request: (AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsStreamingFetchFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1StreamingFetchFeatureValuesResponse) The response message.
      """
    config = self.GetMethodConfig('StreamingFetchFeatureValues')
    return self._RunMethod(config, request, global_params=global_params)