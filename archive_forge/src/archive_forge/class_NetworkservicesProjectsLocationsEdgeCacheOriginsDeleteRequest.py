from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheOriginsDeleteRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheOriginsDeleteRequest object.

  Fields:
    name: Required. The name of the EdgeCacheOrigin to delete. Must be in the
      format `projects/*/locations/global/edgeCacheOrigins/*`.
  """
    name = _messages.StringField(1, required=True)