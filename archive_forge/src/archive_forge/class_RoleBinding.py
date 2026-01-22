from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoleBinding(_messages.Message):
    """IAM Role bindings that will be created on a successful grant.

  Fields:
    conditionExpression: Optional. The expression field of the IAM condition
      to be associated with the role. If specified, a user with an active
      grant for this entitlement would be able to access the resource only if
      this condition evaluates to true for their request. This field uses the
      same CEL format as that of IAM and supports all attributes that IAM
      supports, except tags. https://cloud.google.com/iam/docs/conditions-
      overview#attributes.
    role: Required. IAM role to be granted.
      https://cloud.google.com/iam/docs/roles-overview.
  """
    conditionExpression = _messages.StringField(1)
    role = _messages.StringField(2)