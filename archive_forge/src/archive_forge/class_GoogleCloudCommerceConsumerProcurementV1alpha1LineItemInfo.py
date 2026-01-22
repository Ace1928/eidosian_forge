from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudCommerceConsumerProcurementV1alpha1LineItemInfo(_messages.Message):
    """Line item information.

  Messages:
    SystemPropertiesValue: Output only. System provided key value pairs.

  Fields:
    addOnDetails: Output only. Add on information of a line item, if
      applicable.
    customPricing: Output only. The custom pricing information for this line
      item, if applicable.
    entitlementInfo: Output only. Entitlement info associated with this line
      item.
    flavorExternalName: External name of the flavor being purchased.
    offer: Optional. The name of the offer can have either of these formats:
      'billingAccounts/{billing_account}/offers/{offer}', or
      'services/{service}/standardOffers/{offer}'.
    parameters: Optional. User-provided parameters.
    productExternalName: External name of the product being purchased.
    quoteExternalName: Output only. External name of the quote this product is
      associated with. Present if the product is part of a Quote.
    subscription: Output only. Information about the subscription created, if
      applicable.
    systemProperties: Output only. System provided key value pairs.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SystemPropertiesValue(_messages.Message):
        """Output only. System provided key value pairs.

    Messages:
      AdditionalProperty: An additional property for a SystemPropertiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        SystemPropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SystemPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    addOnDetails = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1AddOnDetails', 1)
    customPricing = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1CustomPricing', 2)
    entitlementInfo = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1EntitlementInfo', 3)
    flavorExternalName = _messages.StringField(4)
    offer = _messages.StringField(5)
    parameters = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Parameter', 6, repeated=True)
    productExternalName = _messages.StringField(7)
    quoteExternalName = _messages.StringField(8)
    subscription = _messages.MessageField('GoogleCloudCommerceConsumerProcurementV1alpha1Subscription', 9)
    systemProperties = _messages.MessageField('SystemPropertiesValue', 10)