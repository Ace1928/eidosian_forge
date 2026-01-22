from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudcommerceconsumerprocurementBillingAccountsOrdersEventsListRequest(_messages.Message):
    """A CloudcommerceconsumerprocurementBillingAccountsOrdersEventsListRequest
  object.

  Fields:
    pageSize: The maximum number of entries requested. The default page size
      is 25 and the maximum page size is 200.
    pageToken: The token for fetching the next page.
    parent: Required. The parent resource to request for events. This field
      has the format 'billingAccounts/{billing-account-id}/orders/{order-id}'.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)