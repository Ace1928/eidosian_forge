from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def MergeVersionAliases(self, request, global_params=None):
    """Merges a set of aliases for a Model version.

      Args:
        request: (AiplatformProjectsLocationsModelsMergeVersionAliasesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1Model) The response message.
      """
    config = self.GetMethodConfig('MergeVersionAliases')
    return self._RunMethod(config, request, global_params=global_params)