from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleAccount(_messages.Message):
    """Describes authentication configuration that uses a Google account.

  Fields:
    password: Required. Input only. The password of the Google account. The
      credential is stored encrypted and not returned in any response nor
      included in audit logs.
    username: Required. The user name of the Google account.
  """
    password = _messages.StringField(1)
    username = _messages.StringField(2)