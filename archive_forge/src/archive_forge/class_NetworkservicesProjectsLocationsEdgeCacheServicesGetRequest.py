from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheServicesGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheServicesGetRequest object.

  Fields:
    name: Required. The name of the EdgeCacheService resource to get. Must be
      in the format `projects/*/locations/global/edgeCacheServices/*`.
  """
    name = _messages.StringField(1, required=True)