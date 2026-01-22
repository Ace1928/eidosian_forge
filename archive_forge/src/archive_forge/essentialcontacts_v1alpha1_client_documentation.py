from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.essentialcontacts.v1alpha1 import essentialcontacts_v1alpha1_messages as messages
Verifies the email of a contact. This method does not require any authorization; authorization is based solely on the validity of the verification_token.

      Args:
        request: (EssentialcontactsProjectsContactsVerifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      