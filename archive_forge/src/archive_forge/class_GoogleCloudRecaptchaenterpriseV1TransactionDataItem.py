from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1TransactionDataItem(_messages.Message):
    """Line items being purchased in this transaction.

  Fields:
    merchantAccountId: Optional. When a merchant is specified, its
      corresponding account_id. Necessary to populate marketplace-style
      transactions.
    name: Optional. The full name of the item.
    quantity: Optional. The quantity of this item that is being purchased.
    value: Optional. The value per item that the user is paying, in the
      transaction currency, after discounts.
  """
    merchantAccountId = _messages.StringField(1)
    name = _messages.StringField(2)
    quantity = _messages.IntegerField(3)
    value = _messages.FloatField(4)