from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsOrdersCancelRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsOrdersCancelRequest
  object.

  Fields:
    googleCloudCommerceConsumerProcurementV1alpha1CancelOrderRequest: A
      GoogleCloudCommerceConsumerProcurementV1alpha1CancelOrderRequest
      resource to be passed as the request body.
    name: Required. The resource name of the order.
  """
    googleCloudCommerceConsumerProcurementV1alpha1CancelOrderRequest = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1CancelOrderRequest', 1)
    name = _messages.StringField(2, required=True)