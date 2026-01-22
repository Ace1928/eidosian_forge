from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
class BillingAccountsAccountsService(base_api.BaseApiService):
    """Service class for the billingAccounts_accounts resource."""
    _NAME = 'billingAccounts_accounts'

    def __init__(self, client):
        super(CloudcommerceconsumerprocurementV1alpha1.BillingAccountsAccountsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Account.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsAccountsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/accounts', http_method='POST', method_id='cloudcommerceconsumerprocurement.billingAccounts.accounts.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/accounts', request_field='googleCloudCommerceConsumerProcurementV1alpha1Account', request_type_name='CloudcommerceconsumerprocurementBillingAccountsAccountsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing Account. An account can only be deleted when there are no orders associated with that account.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsAccountsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/accounts/{accountsId}', http_method='DELETE', method_id='cloudcommerceconsumerprocurement.billingAccounts.accounts.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='CloudcommerceconsumerprocurementBillingAccountsAccountsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the requested Account resource.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsAccountsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1Account) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/accounts/{accountsId}', http_method='GET', method_id='cloudcommerceconsumerprocurement.billingAccounts.accounts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='CloudcommerceconsumerprocurementBillingAccountsAccountsGetRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1Account', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Account resources that the user has access to, within the scope of the parent resource.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsAccountsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1ListAccountsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/accounts', http_method='GET', method_id='cloudcommerceconsumerprocurement.billingAccounts.accounts.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/accounts', request_field='', request_type_name='CloudcommerceconsumerprocurementBillingAccountsAccountsListRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1ListAccountsResponse', supports_download=False)