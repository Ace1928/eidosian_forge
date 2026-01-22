from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
class BillingAccountsOrdersService(base_api.BaseApiService):
    """Service class for the billingAccounts_orders resource."""
    _NAME = 'billingAccounts_orders'

    def __init__(self, client):
        super(CloudcommerceconsumerprocurementV1alpha1.BillingAccountsOrdersService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels an existing Order. Every product procured in the Order will be cancelled.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/orders/{ordersId}:cancel', http_method='POST', method_id='cloudcommerceconsumerprocurement.billingAccounts.orders.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:cancel', request_field='googleCloudCommerceConsumerProcurementV1alpha1CancelOrderRequest', request_type_name='CloudcommerceconsumerprocurementBillingAccountsOrdersCancelRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the requested Order resource.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1Order) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/orders/{ordersId}', http_method='GET', method_id='cloudcommerceconsumerprocurement.billingAccounts.orders.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='CloudcommerceconsumerprocurementBillingAccountsOrdersGetRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1Order', supports_download=False)

    def GetAuditLog(self, request, global_params=None):
        """Returns the requested AuditLog resource. To be deprecated.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersGetAuditLogRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1AuditLog) The response message.
      """
        config = self.GetMethodConfig('GetAuditLog')
        return self._RunMethod(config, request, global_params=global_params)
    GetAuditLog.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/orders/{ordersId}/auditLog', http_method='GET', method_id='cloudcommerceconsumerprocurement.billingAccounts.orders.getAuditLog', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='CloudcommerceconsumerprocurementBillingAccountsOrdersGetAuditLogRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1AuditLog', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Order resources that the user has access to, within the scope of the parent resource.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudCommerceConsumerProcurementV1alpha1ListOrdersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/orders', http_method='GET', method_id='cloudcommerceconsumerprocurement.billingAccounts.orders.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/orders', request_field='', request_type_name='CloudcommerceconsumerprocurementBillingAccountsOrdersListRequest', response_type_name='GoogleCloudCommerceConsumerProcurementV1alpha1ListOrdersResponse', supports_download=False)

    def Modify(self, request, global_params=None):
        """Modifies an existing Order resource.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersModifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Modify')
        return self._RunMethod(config, request, global_params=global_params)
    Modify.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/orders/{ordersId}:modify', http_method='POST', method_id='cloudcommerceconsumerprocurement.billingAccounts.orders.modify', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}:modify', request_field='googleCloudCommerceConsumerProcurementV1alpha1ModifyOrderRequest', request_type_name='CloudcommerceconsumerprocurementBillingAccountsOrdersModifyRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Place(self, request, global_params=None):
        """Creates a new Order. This API only supports GCP spend-based committed use discounts specified by GCP documentation. The returned long-running operation is in-progress until the backend completes the creation of the resource. Once completed, the order is in OrderState.ORDER_STATE_ACTIVE. In case of failure, the order resource will be removed.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersPlaceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Place')
        return self._RunMethod(config, request, global_params=global_params)
    Place.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/orders:place', http_method='POST', method_id='cloudcommerceconsumerprocurement.billingAccounts.orders.place', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/orders:place', request_field='googleCloudCommerceConsumerProcurementV1alpha1PlaceOrderRequest', request_type_name='CloudcommerceconsumerprocurementBillingAccountsOrdersPlaceRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)