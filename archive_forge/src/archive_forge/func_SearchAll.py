from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1p1beta1 import cloudasset_v1p1beta1_messages as messages
def SearchAll(self, request, global_params=None):
    """Searches all the resources within a given accessible CRM scope (project/folder/organization). This RPC gives callers especially administrators the ability to search all the resources within a scope, even if they don't have `.get` permission of all the resources. Callers should have `cloud.assets.SearchAllResources` permission on the requested scope, otherwise the request will be rejected.

      Args:
        request: (CloudassetResourcesSearchAllRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchAllResourcesResponse) The response message.
      """
    config = self.GetMethodConfig('SearchAll')
    return self._RunMethod(config, request, global_params=global_params)