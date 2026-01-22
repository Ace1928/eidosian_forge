from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheKeysetsDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheKeysetsDeleteRequest object.

  Fields:
    name: Required. The name of the EdgeCacheKeyset resource to delete. Must
      be in the format `projects/*/locations/global/edgeCacheKeysets/*`.
  """
    name = _messages.StringField(1, required=True)