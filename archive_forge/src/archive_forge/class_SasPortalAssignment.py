from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalAssignment(_messages.Message):
    """Associates `members` with a `role`.

  Fields:
    members: The identities the role is assigned to. It can have the following
      values: * `{user_email}`: An email address that represents a specific
      Google account. For example: `alice@gmail.com`. * `{group_email}`: An
      email address that represents a Google group. For example,
      `viewers@gmail.com`.
    role: Required. Role that is assigned to `members`.
  """
    members = _messages.StringField(1, repeated=True)
    role = _messages.StringField(2)