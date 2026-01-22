from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
class BillingAccountsOrdersOperationsService(base_api.BaseApiService):
    """Service class for the billingAccounts_orders_operations resource."""
    _NAME = 'billingAccounts_orders_operations'

    def __init__(self, client):
        super(CloudcommerceconsumerprocurementV1alpha1.BillingAccountsOrdersOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (CloudcommerceconsumerprocurementBillingAccountsOrdersOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/billingAccounts/{billingAccountsId}/orders/{ordersId}/operations/{operationsId}', http_method='GET', method_id='cloudcommerceconsumerprocurement.billingAccounts.orders.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='CloudcommerceconsumerprocurementBillingAccountsOrdersOperationsGetRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)