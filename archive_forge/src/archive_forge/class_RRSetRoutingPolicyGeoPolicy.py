from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RRSetRoutingPolicyGeoPolicy(_messages.Message):
    """Configures a RRSetRoutingPolicy that routes based on the geo location of
  the querying user.

  Fields:
    enableFencing: Without fencing, if health check fails for all configured
      items in the current geo bucket, we failover to the next nearest geo
      bucket. With fencing, if health checking is enabled, as long as some
      targets in the current geo bucket are healthy, we return only the
      healthy targets. However, if all targets are unhealthy, we don't
      failover to the next nearest bucket; instead, we return all the items in
      the current bucket even when all targets are unhealthy.
    items: The primary geo routing configuration. If there are multiple items
      with the same location, an error is returned instead.
    kind: A string attribute.
  """
    enableFencing = _messages.BooleanField(1)
    items = _messages.MessageField('RRSetRoutingPolicyGeoPolicyGeoPolicyItem', 2, repeated=True)
    kind = _messages.StringField(3, default='dns#rRSetRoutingPolicyGeoPolicy')