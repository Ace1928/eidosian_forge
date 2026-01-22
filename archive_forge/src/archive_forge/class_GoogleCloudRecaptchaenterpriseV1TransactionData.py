from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1TransactionData(_messages.Message):
    """Transaction data associated with a payment protected by reCAPTCHA
  Enterprise.

  Fields:
    billingAddress: Optional. Address associated with the payment method when
      applicable.
    cardBin: Optional. The Bank Identification Number - generally the first 6
      or 8 digits of the card.
    cardLastFour: Optional. The last four digits of the card.
    currencyCode: Optional. The currency code in ISO-4217 format.
    gatewayInfo: Optional. Information about the payment gateway's response to
      the transaction.
    items: Optional. Items purchased in this transaction.
    merchants: Optional. Information about the user or users fulfilling the
      transaction.
    paymentMethod: Optional. The payment method for the transaction. The
      allowed values are: * credit-card * debit-card * gift-card *
      processor-{name} (If a third-party is used, for example, processor-
      paypal) * custom-{name} (If an alternative method is used, for example,
      custom-crypto)
    shippingAddress: Optional. Destination address if this transaction
      involves shipping a physical item.
    shippingValue: Optional. The value of shipping in the specified currency.
      0 for free or no shipping.
    transactionId: Unique identifier for the transaction. This custom
      identifier can be used to reference this transaction in the future, for
      example, labeling a refund or chargeback event. Two attempts at the same
      transaction should use the same transaction id.
    user: Optional. Information about the user paying/initiating the
      transaction.
    value: Optional. The decimal value of the transaction in the specified
      currency.
  """
    billingAddress = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1TransactionDataAddress', 1)
    cardBin = _messages.StringField(2)
    cardLastFour = _messages.StringField(3)
    currencyCode = _messages.StringField(4)
    gatewayInfo = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1TransactionDataGatewayInfo', 5)
    items = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1TransactionDataItem', 6, repeated=True)
    merchants = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1TransactionDataUser', 7, repeated=True)
    paymentMethod = _messages.StringField(8)
    shippingAddress = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1TransactionDataAddress', 9)
    shippingValue = _messages.FloatField(10)
    transactionId = _messages.StringField(11)
    user = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1TransactionDataUser', 12)
    value = _messages.FloatField(13)