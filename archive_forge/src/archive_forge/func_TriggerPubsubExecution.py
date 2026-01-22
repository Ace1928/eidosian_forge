from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workflowexecutions.v1 import workflowexecutions_v1_messages as messages
def TriggerPubsubExecution(self, request, global_params=None):
    """Triggers a new execution using the latest revision of the given workflow by a Pub/Sub push notification.

      Args:
        request: (WorkflowexecutionsProjectsLocationsWorkflowsTriggerPubsubExecutionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Execution) The response message.
      """
    config = self.GetMethodConfig('TriggerPubsubExecution')
    return self._RunMethod(config, request, global_params=global_params)