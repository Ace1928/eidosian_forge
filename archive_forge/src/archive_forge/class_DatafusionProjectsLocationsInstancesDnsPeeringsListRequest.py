from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsInstancesDnsPeeringsListRequest(_messages.Message):
    """A DatafusionProjectsLocationsInstancesDnsPeeringsListRequest object.

  Fields:
    pageSize: The maximum number of dns peerings to return. The service may
      return fewer than this value. If unspecified, at most 50 dns peerings
      will be returned. The maximum value is 200; values above 200 will be
      coerced to 200.
    pageToken: A page token, received from a previous `ListDnsPeerings` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListDnsPeerings` must match the call that
      provided the page token.
    parent: Required. The parent, which owns this collection of dns peerings.
      Format: projects/{project}/locations/{location}/instances/{instance}
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)