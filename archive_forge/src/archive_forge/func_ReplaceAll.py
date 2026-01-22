from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.accesscontextmanager.v1alpha import accesscontextmanager_v1alpha_messages as messages
def ReplaceAll(self, request, global_params=None):
    """Replace all existing service perimeters in an access policy with the service perimeters provided. This is done atomically. The long-running operation from this RPC has a successful status after all replacements propagate to long-lasting storage. Replacements containing errors result in an error response for the first error encountered. Upon an error, replacement are cancelled and existing service perimeters are not affected. The Operation.response field contains ReplaceServicePerimetersResponse.

      Args:
        request: (AccesscontextmanagerAccessPoliciesServicePerimetersReplaceAllRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ReplaceAll')
    return self._RunMethod(config, request, global_params=global_params)