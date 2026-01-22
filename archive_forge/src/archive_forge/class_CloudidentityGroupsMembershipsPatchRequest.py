from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsMembershipsPatchRequest(_messages.Message):
    """A CloudidentityGroupsMembershipsPatchRequest object.

  Fields:
    membership: A Membership resource to be passed as the request body.
    name: Output only. The [resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      `Membership`. Shall be of the form
      `groups/{group_id}/memberships/{membership_id}`.
    updateMask: Required. The fully-qualified names of fields to update.
  """
    membership = _messages.MessageField('Membership', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)