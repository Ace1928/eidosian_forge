from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExpiryDetail(_messages.Message):
    """The `MembershipRole` expiry details.

  Fields:
    expireTime: The time at which the `MembershipRole` will expire.
  """
    expireTime = _messages.StringField(1)