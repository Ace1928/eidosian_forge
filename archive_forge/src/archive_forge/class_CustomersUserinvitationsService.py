from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1 import cloudidentity_v1_messages as messages
class CustomersUserinvitationsService(base_api.BaseApiService):
    """Service class for the customers_userinvitations resource."""
    _NAME = 'customers_userinvitations'

    def __init__(self, client):
        super(CloudidentityV1.CustomersUserinvitationsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels a UserInvitation that was already sent.

      Args:
        request: (CloudidentityCustomersUserinvitationsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/customers/{customersId}/userinvitations/{userinvitationsId}:cancel', http_method='POST', method_id='cloudidentity.customers.userinvitations.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='cancelUserInvitationRequest', request_type_name='CloudidentityCustomersUserinvitationsCancelRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a UserInvitation resource. **Note:** New consumer accounts with the customer's verified domain created within the previous 48 hours will not appear in the result. This delay also applies to newly-verified domains.

      Args:
        request: (CloudidentityCustomersUserinvitationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UserInvitation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/customers/{customersId}/userinvitations/{userinvitationsId}', http_method='GET', method_id='cloudidentity.customers.userinvitations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='CloudidentityCustomersUserinvitationsGetRequest', response_type_name='UserInvitation', supports_download=False)

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
    IsInvitableUser.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/customers/{customersId}/userinvitations/{userinvitationsId}:isInvitableUser', http_method='GET', method_id='cloudidentity.customers.userinvitations.isInvitableUser', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:isInvitableUser', request_field='', request_type_name='CloudidentityCustomersUserinvitationsIsInvitableUserRequest', response_type_name='IsInvitableUserResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of UserInvitation resources. **Note:** New consumer accounts with the customer's verified domain created within the previous 48 hours will not appear in the result. This delay also applies to newly-verified domains.

      Args:
        request: (CloudidentityCustomersUserinvitationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUserInvitationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/customers/{customersId}/userinvitations', http_method='GET', method_id='cloudidentity.customers.userinvitations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/userinvitations', request_field='', request_type_name='CloudidentityCustomersUserinvitationsListRequest', response_type_name='ListUserInvitationsResponse', supports_download=False)

    def Send(self, request, global_params=None):
        """Sends a UserInvitation to email. If the `UserInvitation` does not exist for this request and it is a valid request, the request creates a `UserInvitation`. **Note:** The `get` and `list` methods have a 48-hour delay where newly-created consumer accounts will not appear in the results. You can still send a `UserInvitation` to those accounts if you know the unmanaged email address and IsInvitableUser==True.

      Args:
        request: (CloudidentityCustomersUserinvitationsSendRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Send')
        return self._RunMethod(config, request, global_params=global_params)
    Send.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/customers/{customersId}/userinvitations/{userinvitationsId}:send', http_method='POST', method_id='cloudidentity.customers.userinvitations.send', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:send', request_field='sendUserInvitationRequest', request_type_name='CloudidentityCustomersUserinvitationsSendRequest', response_type_name='Operation', supports_download=False)