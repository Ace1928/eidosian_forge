from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1PlaceOrderRequest(_messages.Message):
    """Request message for ConsumerProcurementService.PlaceOrder.

  Enums:
    AutoRenewalBehaviorValueValuesEnum: Optional. Auto renewal behavior of the
      subscription associated with the order.

  Fields:
    account: The resource name of the account that this order is based on. If
      this field is not specified and the creation of any products in the
      order requires an account, system will look for existing account and
      auto create one if there is no existing one.
    autoRenewalBehavior: Optional. Auto renewal behavior of the subscription
      associated with the order.
    displayName: Required. The user-specified name of the order being placed.
    lineItemInfo: Optional. Places order for offer. Required when an offer-
      based order is being placed.
    placeProductsOrderRequest: Optional. Places order for non-quote products.
    placeQuoteOrderRequest: Optional. Places order for quote.
    provider: Required. Provider of the items being purchased. Provider has
      the format of `providers/{provider_id}`. Optional when an offer is
      specified. TODO(b/241564581) Hide provider id in the consumer API.
    requestId: Optional. A unique identifier for this request. The server will
      ignore subsequent requests that provide a duplicate request ID for at
      least 120 minutes after the first request. The request ID must be a
      valid [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier
      #Format).
    testConfig: Optional. Test configuration for the to-be-placed order.
      Placing test order is only allowed if the parent is a testing billing
      account for the service.
  """

    class AutoRenewalBehaviorValueValuesEnum(_messages.Enum):
        """Optional. Auto renewal behavior of the subscription associated with
    the order.

    Values:
      AUTO_RENEWAL_BEHAVIOR_UNSPECIFIED: If unspecified, the auto renewal
        behavior will follow the default config.
      AUTO_RENEWAL_BEHAVIOR_ENABLE: Auto Renewal will be enabled on
        subscription.
      AUTO_RENEWAL_BEHAVIOR_DISABLE: Auto Renewal will be disabled on
        subscription.
    """
        AUTO_RENEWAL_BEHAVIOR_UNSPECIFIED = 0
        AUTO_RENEWAL_BEHAVIOR_ENABLE = 1
        AUTO_RENEWAL_BEHAVIOR_DISABLE = 2
    account = _messages.StringField(1)
    autoRenewalBehavior = _messages.EnumField('AutoRenewalBehaviorValueValuesEnum', 2)
    displayName = _messages.StringField(3)
    lineItemInfo = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1LineItemInfo', 4, repeated=True)
    placeProductsOrderRequest = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1PlaceProductsOrderRequest', 5)
    placeQuoteOrderRequest = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1PlaceQuoteOrderRequest', 6)
    provider = _messages.StringField(7)
    requestId = _messages.StringField(8)
    testConfig = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1TestConfig', 9)