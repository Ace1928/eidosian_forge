from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
def AnalyzeOrgPolicyGovernedResources(self, request, global_params=None):
    """Analyzes organization policies governed resources under a scope. This RPC only returns resources of types [supported by search APIs](https://cloud.google.com/asset-inventory/docs/supported-asset-types).

      Args:
        request: (CloudassetAnalyzeOrgPolicyGovernedResourcesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeOrgPolicyGovernedResourcesResponse) The response message.
      """
    config = self.GetMethodConfig('AnalyzeOrgPolicyGovernedResources')
    return self._RunMethod(config, request, global_params=global_params)