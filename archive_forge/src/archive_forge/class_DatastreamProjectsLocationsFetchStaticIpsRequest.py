from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastreamProjectsLocationsFetchStaticIpsRequest(_messages.Message):
    """A DatastreamProjectsLocationsFetchStaticIpsRequest object.

  Fields:
    name: Required. The resource name for the location for which static IPs
      should be returned. Must be in the format `projects/*/locations/*`.
    pageSize: Maximum number of Ips to return, will likely not be specified.
    pageToken: A page token, received from a previous `ListStaticIps` call.
      will likely not be specified.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)