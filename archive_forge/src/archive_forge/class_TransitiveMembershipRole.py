from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransitiveMembershipRole(_messages.Message):
    """Message representing the role of a TransitiveMembership.

  Fields:
    role: TransitiveMembershipRole in string format. Currently supported
      TransitiveMembershipRoles: `"MEMBER"`, `"OWNER"`, and `"MANAGER"`.
  """
    role = _messages.StringField(1)