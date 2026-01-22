from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingBillingAccountsPatchRequest(_messages.Message):
    """A CloudbillingBillingAccountsPatchRequest object.

  Fields:
    billingAccount: A BillingAccount resource to be passed as the request
      body.
    name: Required. The name of the billing account resource to be updated.
    updateMask: The update mask applied to the resource. Only "display_name"
      is currently supported.
  """
    billingAccount = _messages.MessageField('BillingAccount', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)