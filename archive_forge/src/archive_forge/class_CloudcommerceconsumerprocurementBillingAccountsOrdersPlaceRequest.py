from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsOrdersPlaceRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsOrdersPlaceRequest
  object.

  Fields:
    googleCloudCommerceConsumerProcurementV1alpha1PlaceOrderRequest: A
      GoogleCloudCommerceConsumerProcurementV1alpha1PlaceOrderRequest resource
      to be passed as the request body.
    parent: Required. The resource name of the parent resource. This field has
      the form `billingAccounts/{billing-account-id}`.
  """
    googleCloudCommerceConsumerProcurementV1alpha1PlaceOrderRequest = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1PlaceOrderRequest', 1)
    parent = _messages.StringField(2, required=True)