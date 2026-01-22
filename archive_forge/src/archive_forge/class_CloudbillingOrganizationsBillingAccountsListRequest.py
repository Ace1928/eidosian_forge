from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingOrganizationsBillingAccountsListRequest(_messages.Message):
    """A CloudbillingOrganizationsBillingAccountsListRequest object.

  Fields:
    filter: Options for how to filter the returned billing accounts. This only
      supports filtering for
      [subaccounts](https://cloud.google.com/billing/docs/concepts) under a
      single provided parent billing account. (for example,
      `master_billing_account=billingAccounts/012345-678901-ABCDEF`). Boolean
      algebra and other fields are not currently supported.
    pageSize: Requested page size. The maximum page size is 100; this is also
      the default.
    pageToken: A token identifying a page of results to return. This should be
      a `next_page_token` value returned from a previous `ListBillingAccounts`
      call. If unspecified, the first page of results is returned.
    parent: Optional. The parent resource to list billing accounts from.
      Format: - `organizations/{organization_id}`, for example,
      `organizations/12345678` - `billingAccounts/{billing_account_id}`, for
      example, `billingAccounts/012345-567890-ABCDEF`
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)