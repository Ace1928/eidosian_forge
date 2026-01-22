from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudshellUsersEnvironmentsPublicKeysCreateRequest(_messages.Message):
    """A CloudshellUsersEnvironmentsPublicKeysCreateRequest object.

  Fields:
    createPublicKeyRequest: A CreatePublicKeyRequest resource to be passed as
      the request body.
    parent: Parent resource name, e.g. `users/me/environments/default`.
  """
    createPublicKeyRequest = _messages.MessageField('CreatePublicKeyRequest', 1)
    parent = _messages.StringField(2, required=True)