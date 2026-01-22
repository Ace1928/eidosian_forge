from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudshellUsersEnvironmentsPublicKeysDeleteRequest(_messages.Message):
    """A CloudshellUsersEnvironmentsPublicKeysDeleteRequest object.

  Fields:
    name: Name of the resource to be deleted, e.g.
      `users/me/environments/default/publicKeys/my-key`.
  """
    name = _messages.StringField(1, required=True)