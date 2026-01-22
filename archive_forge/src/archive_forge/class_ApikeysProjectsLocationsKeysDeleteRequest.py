from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApikeysProjectsLocationsKeysDeleteRequest(_messages.Message):
    """A ApikeysProjectsLocationsKeysDeleteRequest object.

  Fields:
    etag: Optional. The etag known to the client for the expected state of the
      key. This is to be used for optimistic concurrency.
    name: Required. The resource name of the API key to be deleted.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)