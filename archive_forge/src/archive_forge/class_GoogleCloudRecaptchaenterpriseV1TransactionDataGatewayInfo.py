from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1TransactionDataGatewayInfo(_messages.Message):
    """Details about the transaction from the gateway.

  Fields:
    avsResponseCode: Optional. AVS response code from the gateway (available
      only when reCAPTCHA Enterprise is called after authorization).
    cvvResponseCode: Optional. CVV response code from the gateway (available
      only when reCAPTCHA Enterprise is called after authorization).
    gatewayResponseCode: Optional. Gateway response code describing the state
      of the transaction.
    name: Optional. Name of the gateway service (for example, stripe, square,
      paypal).
  """
    avsResponseCode = _messages.StringField(1)
    cvvResponseCode = _messages.StringField(2)
    gatewayResponseCode = _messages.StringField(3)
    name = _messages.StringField(4)