from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1TransactionDataAddress(_messages.Message):
    """Structured address format for billing and shipping addresses.

  Fields:
    address: Optional. The first lines of the address. The first line
      generally contains the street name and number, and further lines may
      include information such as an apartment number.
    administrativeArea: Optional. The state, province, or otherwise
      administrative area of the address.
    locality: Optional. The town/city of the address.
    postalCode: Optional. The postal or ZIP code of the address.
    recipient: Optional. The recipient name, potentially including information
      such as "care of".
    regionCode: Optional. The CLDR country/region of the address.
  """
    address = _messages.StringField(1, repeated=True)
    administrativeArea = _messages.StringField(2)
    locality = _messages.StringField(3)
    postalCode = _messages.StringField(4)
    recipient = _messages.StringField(5)
    regionCode = _messages.StringField(6)