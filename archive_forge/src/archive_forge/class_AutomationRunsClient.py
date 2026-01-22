from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.clouddeploy import client_util
class AutomationRunsClient(object):
    """Client for automation run service in the Cloud Deploy API."""

    def __init__(self, client=None, messages=None):
        """Initialize an automation_run.AutomationRunsClient.

    Args:
      client: base_api.BaseApiClient, the client class for Cloud Deploy.
      messages: module containing the definitions of messages for Cloud Deploy.
    """
        self.client = client or client_util.GetClientInstance()
        self.messages = messages or client_util.GetMessagesModule(client)
        self._service = self.client.projects_locations_deliveryPipelines_automationRuns

    def Cancel(self, name):
        """Cancels an automation run.

    Args:
      name: Name of the AutomationRun. Format is
        projects/{project}/locations/{location}/deliveryPipelines/{deliveryPipeline}/automationRuns/{automationRun}.

    Returns:
      CancelAutomationRunResponse message.
    """
        request = self.messages.ClouddeployProjectsLocationsDeliveryPipelinesAutomationRunsCancelRequest(name=name)
        return self._service.Cancel(request)