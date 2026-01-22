from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FetchReadTokenResponse(_messages.Message):
    """Message for responding to get read token.

  Fields:
    expirationTime: Expiration timestamp. Can be empty if unknown or non-
      expiring.
    token: The token content.
  """
    expirationTime = _messages.StringField(1)
    token = _messages.StringField(2)