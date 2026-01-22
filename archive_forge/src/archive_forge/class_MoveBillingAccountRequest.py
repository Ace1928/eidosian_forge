from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MoveBillingAccountRequest(_messages.Message):
    """Request message for `MoveBillingAccount` RPC.

  Fields:
    destinationParent: Required. The resource name of the Organization to move
      the billing account under. Must be of the form
      `organizations/{organization_id}`.
  """
    destinationParent = _messages.StringField(1)