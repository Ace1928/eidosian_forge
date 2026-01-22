from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheOriginsListRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheOriginsListRequest object.

  Fields:
    pageSize: The maximum number of EdgeCacheOrigin resources to return per
      call.
    pageToken: The value returned by the last `ListEdgeCacheOriginsResponse`
      resource. Indicates that this is a continuation of a previous
      `ListEdgeCacheOrigins` call, and that the system can return the next
      page of data.
    parent: Required. The project and location to list EdgeCacheOrigin
      resources from, specified in the format `projects/*/locations/global`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)