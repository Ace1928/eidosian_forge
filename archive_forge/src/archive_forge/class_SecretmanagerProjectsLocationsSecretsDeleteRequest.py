from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretmanagerProjectsLocationsSecretsDeleteRequest(_messages.Message):
    """A SecretmanagerProjectsLocationsSecretsDeleteRequest object.

  Fields:
    etag: Optional. Etag of the Secret. The request succeeds if it matches the
      etag of the currently stored secret object. If the etag is omitted, the
      request succeeds.
    name: Required. The resource name of the Secret to delete in the format
      `projects/*/secrets/*`.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)