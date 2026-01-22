from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListBillingAccountsResponse(_messages.Message):
    """Response message for `ListBillingAccounts`.

  Fields:
    billingAccounts: A list of billing accounts.
    nextPageToken: A token to retrieve the next page of results. To retrieve
      the next page, call `ListBillingAccounts` again with the `page_token`
      field set to this value. This field is empty if there are no more
      results to retrieve.
  """
    billingAccounts = _messages.MessageField('BillingAccount', 1, repeated=True)
    nextPageToken = _messages.StringField(2)