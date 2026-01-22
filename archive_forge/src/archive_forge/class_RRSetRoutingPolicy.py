from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RRSetRoutingPolicy(_messages.Message):
    """A RRSetRoutingPolicy represents ResourceRecordSet data that is returned
  dynamically with the response varying based on configured properties such as
  geolocation or by weighted random selection.

  Fields:
    geo: A RRSetRoutingPolicyGeoPolicy attribute.
    geoPolicy: A RRSetRoutingPolicyGeoPolicy attribute.
    healthCheck: The selfLink attribute of the HealthCheck resource to use for
      this RRSetRoutingPolicy.
      https://cloud.google.com/compute/docs/reference/rest/v1/healthChecks
    kind: A string attribute.
    primaryBackup: A RRSetRoutingPolicyPrimaryBackupPolicy attribute.
    wrr: A RRSetRoutingPolicyWrrPolicy attribute.
    wrrPolicy: A RRSetRoutingPolicyWrrPolicy attribute.
  """
    geo = _messages.MessageField('RRSetRoutingPolicyGeoPolicy', 1)
    geoPolicy = _messages.MessageField('RRSetRoutingPolicyGeoPolicy', 2)
    healthCheck = _messages.StringField(3)
    kind = _messages.StringField(4, default='dns#rRSetRoutingPolicy')
    primaryBackup = _messages.MessageField('RRSetRoutingPolicyPrimaryBackupPolicy', 5)
    wrr = _messages.MessageField('RRSetRoutingPolicyWrrPolicy', 6)
    wrrPolicy = _messages.MessageField('RRSetRoutingPolicyWrrPolicy', 7)