from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def AddContextChildren(self, request, global_params=None):
    """Adds a set of Contexts as children to a parent Context. If any of the child Contexts have already been added to the parent Context, they are simply skipped. If this call would create a cycle or cause any Context to have more than 10 parents, the request will fail with an INVALID_ARGUMENT error.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsAddContextChildrenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1AddContextChildrenResponse) The response message.
      """
    config = self.GetMethodConfig('AddContextChildren')
    return self._RunMethod(config, request, global_params=global_params)