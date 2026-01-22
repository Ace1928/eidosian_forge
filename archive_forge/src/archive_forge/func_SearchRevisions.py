from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.telcoautomation.v1 import telcoautomation_v1_messages as messages
def SearchRevisions(self, request, global_params=None):
    """Searches across deployment revisions.

      Args:
        request: (TelcoautomationProjectsLocationsOrchestrationClustersDeploymentsSearchRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchDeploymentRevisionsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchRevisions')
    return self._RunMethod(config, request, global_params=global_params)