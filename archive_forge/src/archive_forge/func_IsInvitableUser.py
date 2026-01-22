from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
def IsInvitableUser(self, request, global_params=None):
    """Verifies whether a user account is eligible to receive a UserInvitation (is an unmanaged account). Eligibility is based on the following criteria: * the email address is a consumer account and it's the primary email address of the account, and * the domain of the email address matches an existing verified Google Workspace or Cloud Identity domain If both conditions are met, the user is eligible. **Note:** This method is not supported for Workspace Essentials customers.

      Args:
        request: (CloudidentityCustomersUserinvitationsIsInvitableUserRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IsInvitableUserResponse) The response message.
      """
    config = self.GetMethodConfig('IsInvitableUser')
    return self._RunMethod(config, request, global_params=global_params)