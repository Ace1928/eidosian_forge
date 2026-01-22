from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DestroySecretVersionRequest(_messages.Message):
    """Request message for SecretManagerService.DestroySecretVersion.

  Fields:
    etag: Optional. Etag of the SecretVersion. The request succeeds if it
      matches the etag of the currently stored secret version object. If the
      etag is omitted, the request succeeds.
  """
    etag = _messages.StringField(1)