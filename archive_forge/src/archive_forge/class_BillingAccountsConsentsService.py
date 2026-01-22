from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
class BillingAccountsConsentsService(base_api.BaseApiService):
    """Service class for the billingAccounts_consents resource."""
    _NAME = 'billingAccounts_consents'

    def __init__(self, client):
        super(CloudcommerceconsumerprocurementV1alpha1.BillingAccountsConsentsService, self).__init__(client)
        self._upload_configs = {}

    def Check(self, request, global_params=None):
        """Checks if a customer's consents satisfy the current agreement.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsConsentsCheckRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1CheckConsentResponse) The response message.
      """
        config = self.GetMethodConfig('Check')
        return self._RunMethod(config, request, global_params=global_params)
    Check.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/consents:check', http_method='POST', method_id='cloudcommerceconsumerprocurement.billingAccounts.consents.check', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/consents:check', request_field='googleCloudCommerceConsumerProcurementV1alpha1CheckConsentRequest', request_type_name='CloudcommerceconsumerprocurementBillingAccountsConsentsCheckRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1CheckConsentResponse', supports_download=False)

    def Grant(self, request, global_params=None):
        """Grants consent.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsConsentsGrantRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1Consent) The response message.
      """
        config = self.GetMethodConfig('Grant')
        return self._RunMethod(config, request, global_params=global_params)
    Grant.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/consents:grant', http_method='POST', method_id='cloudcommerceconsumerprocurement.billingAccounts.consents.grant', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/consents:grant', request_field='googleCloudCommerceConsumerProcurementV1alpha1GrantConsentRequest', request_type_name='CloudcommerceconsumerprocurementBillingAccountsConsentsGrantRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1Consent', supports_download=False)

    def List(self, request, global_params=None):
        """Lists current consents.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsConsentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1ListConsentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/consents', http_method='GET', method_id='cloudcommerceconsumerprocurement.billingAccounts.consents.list', ordered_params=['parent'], path_params=['parent'], query_params=['agreement', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/consents', request_field='', request_type_name='CloudcommerceconsumerprocurementBillingAccountsConsentsListRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1ListConsentsResponse', supports_download=False)

    def Revoke(self, request, global_params=None):
        """Revokes a consent. Revocation is only allowed on a revokable agreement with a current Consent.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsConsentsRevokeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1Consent) The response message.
      """
        config = self.GetMethodConfig('Revoke')
        return self._RunMethod(config, request, global_params=global_params)
    Revoke.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/consents/{consentsId}:revoke', http_method='POST', method_id='cloudcommerceconsumerprocurement.billingAccounts.consents.revoke', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:revoke', request_field='googleCloudCommerceConsumerProcurementV1alpha1RevokeConsentRequest', request_type_name='CloudcommerceconsumerprocurementBillingAccountsConsentsRevokeRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1Consent', supports_download=False)