from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def AddExecutionEvents(self, request, global_params=None):
    """Adds Events to the specified Execution. An Event indicates whether an Artifact was used as an input or output for an Execution. If an Event already exists between the Execution and the Artifact, the Event is skipped.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresExecutionsAddExecutionEventsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1AddExecutionEventsResponse) The response message.
      """
    config = self.GetMethodConfig('AddExecutionEvents')
    return self._RunMethod(config, request, global_params=global_params)