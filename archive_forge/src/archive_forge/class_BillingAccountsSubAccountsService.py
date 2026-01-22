from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbilling.v1 import cloudbilling_v1_messages as messages
class BillingAccountsSubAccountsService(base_api.BaseApiService):
    """Service class for the billingAccounts_subAccounts resource."""
    _NAME = 'billingAccounts_subAccounts'

    def __init__(self, client):
        super(CloudbillingV1.BillingAccountsSubAccountsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """This method creates [billing subaccounts](https://cloud.google.com/billing/docs/concepts#subaccounts). Google Cloud resellers should use the Channel Services APIs, [accounts.customers.create](https://cloud.google.com/channel/docs/reference/rest/v1/accounts.customers/create) and [accounts.customers.entitlements.create](https://cloud.google.com/channel/docs/reference/rest/v1/accounts.customers.entitlements/create). When creating a subaccount, the current authenticated user must have the `billing.accounts.update` IAM permission on the parent account, which is typically given to billing account [administrators](https://cloud.google.com/billing/docs/how-to/billing-access). This method will return an error if the parent account has not been provisioned for subaccounts.

      Args:
        request: (BillingAccount) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BillingAccount) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/billingAccounts/{billingAccountsId}/subAccounts', http_method='POST', method_id='cloudbilling.billingAccounts.subAccounts.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/subAccounts', request_field='<request>', request_type_name='BillingAccount', response_type_name='BillingAccount', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the billing accounts that the current authenticated user has permission to [view](https://cloud.google.com/billing/docs/how-to/billing-access).

      Args:
        request: (CloudbillingBillingAccountsSubAccountsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBillingAccountsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/billingAccounts/{billingAccountsId}/subAccounts', http_method='GET', method_id='cloudbilling.billingAccounts.subAccounts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/subAccounts', request_field='', request_type_name='CloudbillingBillingAccountsSubAccountsListRequest', response_type_name='ListBillingAccountsResponse', supports_download=False)