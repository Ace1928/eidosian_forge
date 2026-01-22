from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsEdgeCacheOriginsPatchRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsEdgeCacheOriginsPatchRequest object.

  Fields:
    edgeCacheOrigin: A EdgeCacheOrigin resource to be passed as the request
      body.
    name: Required. The name of the resource as provided by the client when
      the resource is created. The name must be 1-64 characters long, and
      match the regular expression `[a-zA-Z]([a-zA-Z0-9_-])*`, which means
      that the first character must be a letter, and all following characters
      must be a dash, an underscore, a letter, or a digit.
    updateMask: Optional. `FieldMask` is used to specify the fields to
      overwrite in the EdgeCacheOrigin resource by the update. The fields
      specified in `update_mask` are relative to the resource, not the full
      request. A field is overwritten if it is in the mask. If the user does
      not provide a mask value, then all fields are overwritten.
  """
    edgeCacheOrigin = _messages.MessageField('EdgeCacheOrigin', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)