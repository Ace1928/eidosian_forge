from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
def AnalyzeOrgPolicyGovernedContainers(self, request, global_params=None):
    """Analyzes organization policies governed containers (projects, folders or organization) under a scope.

      Args:
        request: (CloudassetAnalyzeOrgPolicyGovernedContainersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeOrgPolicyGovernedContainersResponse) The response message.
      """
    config = self.GetMethodConfig('AnalyzeOrgPolicyGovernedContainers')
    return self._RunMethod(config, request, global_params=global_params)