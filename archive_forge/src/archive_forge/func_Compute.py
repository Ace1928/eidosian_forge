from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.essentialcontacts.v1 import essentialcontacts_v1_messages as messages
def Compute(self, request, global_params=None):
    """Lists all contacts for the resource that are subscribed to the specified notification categories, including contacts inherited from any parent resources.

      Args:
        request: (EssentialcontactsProjectsContactsComputeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1ComputeContactsResponse) The response message.
      """
    config = self.GetMethodConfig('Compute')
    return self._RunMethod(config, request, global_params=global_params)