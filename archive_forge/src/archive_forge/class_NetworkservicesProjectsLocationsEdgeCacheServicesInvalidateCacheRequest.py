from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheServicesInvalidateCacheRequest(_messages.Message):
    """A
  NetworkservicesProjectsLocationsEdgeCacheServicesInvalidateCacheRequest
  object.

  Fields:
    edgeCacheService: Required. The name of the EdgeCacheService resource to
      apply the invalidation request to. Must be in the format
      `projects/*/locations/global/edgeCacheServices/*`.
    invalidateCacheRequest: A InvalidateCacheRequest resource to be passed as
      the request body.
  """
    edgeCacheService = _messages.StringField(1, required=True)
    invalidateCacheRequest = _messages.MessageField('InvalidateCacheRequest', 2)