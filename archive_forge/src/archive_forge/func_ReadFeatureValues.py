from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def ReadFeatureValues(self, request, global_params=None):
    """Reads Feature values of a specific entity of an EntityType. For reading feature values of multiple entities of an EntityType, please use StreamingReadFeatureValues.

      Args:
        request: (AiplatformProjectsLocationsFeaturestoresEntityTypesReadFeatureValuesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadFeatureValuesResponse) The response message.
      """
    config = self.GetMethodConfig('ReadFeatureValues')
    return self._RunMethod(config, request, global_params=global_params)