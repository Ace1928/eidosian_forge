from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserInfo(_messages.Message):
    """Information about a user.

  Fields:
    email: E-mail address of the user.
  """
    email = _messages.StringField(1)