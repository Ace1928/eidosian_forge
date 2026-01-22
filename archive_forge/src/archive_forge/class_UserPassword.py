from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserPassword(_messages.Message):
    """The username/password for a database user. Used for specifying initial
  users at cluster creation time.

  Fields:
    password: The initial password for the user.
    passwordSet: Output only. Indicates if the initial_user.password field has
      been set.
    user: The database username.
  """
    password = _messages.StringField(1)
    passwordSet = _messages.BooleanField(2)
    user = _messages.StringField(3)