from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudbillingProjectsUpdateBillingInfoRequest(_messages.Message):
    """A CloudbillingProjectsUpdateBillingInfoRequest object.

  Fields:
    name: Required. The resource name of the project associated with the
      billing information that you want to update. For example,
      `projects/tokyo-rain-123`.
    projectBillingInfo: A ProjectBillingInfo resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    projectBillingInfo = _messages.MessageField('ProjectBillingInfo', 2)