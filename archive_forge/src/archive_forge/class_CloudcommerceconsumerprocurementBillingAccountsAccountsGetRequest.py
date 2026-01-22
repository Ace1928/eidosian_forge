from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsAccountsGetRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsAccountsGetRequest
  object.

  Fields:
    name: Required. The resource name of the account to retrieve.
  """
    name = _messages.StringField(1, required=True)