from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1CreditDeveloperBalanceRequest(_messages.Message):
    """Request for CreditDeveloperBalance.

  Fields:
    transactionAmount: The amount of money to be credited. The wallet
      corresponding to the currency specified within `transaction_amount` will
      be updated. For example, if you specified `currency_code` within
      `transaction_amount` as "USD", then the amount would be added to the
      wallet which has the "USD" currency or if no such wallet exists, a new
      wallet will be created with the "USD" currency.
    transactionId: Each transaction_id uniquely identifies a credit balance
      request. If multiple requests are received with the same transaction_id,
      only one of them will be considered.
  """
    transactionAmount = _messages.MessageField('GoogleTypeMoney', 1)
    transactionId = _messages.StringField(2)