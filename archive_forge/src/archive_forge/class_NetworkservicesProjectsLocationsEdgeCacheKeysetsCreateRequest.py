from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheKeysetsCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheKeysetsCreateRequest object.

  Fields:
    edgeCacheKeyset: A EdgeCacheKeyset resource to be passed as the request
      body.
    edgeCacheKeysetId: Required. The short name of the EdgeCacheKeyset
      resource to create, such as `MyEdgeCacheKeyset`.
    parent: Required. The parent resource of the EdgeCacheKeyset resource.
      Must be in the format `projects/*/locations/global`.
  """
    edgeCacheKeyset = _messages.MessageField('EdgeCacheKeyset', 1)
    edgeCacheKeysetId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)