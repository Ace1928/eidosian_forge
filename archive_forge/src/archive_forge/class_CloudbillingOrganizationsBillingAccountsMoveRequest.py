from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingOrganizationsBillingAccountsMoveRequest(_messages.Message):
    """A CloudbillingOrganizationsBillingAccountsMoveRequest object.

  Fields:
    destinationParent: Required. The resource name of the Organization to move
      the billing account under. Must be of the form
      `organizations/{organization_id}`.
    name: Required. The resource name of the billing account to move. Must be
      of the form `billingAccounts/{billing_account_id}`. The specified
      billing account cannot be a subaccount, since a subaccount always
      belongs to the same organization as its parent account.
  """
    destinationParent = _messages.StringField(1, required=True)
    name = _messages.StringField(2, required=True)