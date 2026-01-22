from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudshellUsersEnvironmentsRemovePublicKeyRequest(_messages.Message):
    """A CloudshellUsersEnvironmentsRemovePublicKeyRequest object.

  Fields:
    environment: Environment this key should be removed from, e.g.
      `users/me/environments/default`.
    removePublicKeyRequest: A RemovePublicKeyRequest resource to be passed as
      the request body.
  """
    environment = _messages.StringField(1, required=True)
    removePublicKeyRequest = _messages.MessageField('RemovePublicKeyRequest', 2)