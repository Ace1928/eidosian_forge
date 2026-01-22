from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudcommerceconsumerprocurement.v1alpha1 import cloudcommerceconsumerprocurement_v1alpha1_messages as messages
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