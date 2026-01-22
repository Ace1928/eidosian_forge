from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.essentialcontacts.v1alpha1 import essentialcontacts_v1alpha1_messages as messages
def Resend(self, request, global_params=None):
    """Allows user to resend verification. This will also update the verification expiration date.

      Args:
        request: (EssentialcontactsProjectsContactsResendRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1alpha1Contact) The response message.
      """
    config = self.GetMethodConfig('Resend')
    return self._RunMethod(config, request, global_params=global_params)