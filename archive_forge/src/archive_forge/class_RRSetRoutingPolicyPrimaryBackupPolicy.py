from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RRSetRoutingPolicyPrimaryBackupPolicy(_messages.Message):
    """Configures a RRSetRoutingPolicy such that all queries are responded with
  the primary_targets if they are healthy. And if all of them are unhealthy,
  then we fallback to a geo localized policy.

  Fields:
    backupGeoTargets: Backup targets provide a regional failover policy for
      the otherwise global primary targets. If serving state is set to BACKUP,
      this policy essentially becomes a geo routing policy.
    kind: A string attribute.
    primaryTargets: Endpoints that are health checked before making the
      routing decision. Unhealthy endpoints are omitted from the results. If
      all endpoints are unhealthy, we serve a response based on the
      backup_geo_targets.
    trickleTraffic: When serving state is PRIMARY, this field provides the
      option of sending a small percentage of the traffic to the backup
      targets.
  """
    backupGeoTargets = _messages.MessageField('RRSetRoutingPolicyGeoPolicy', 1)
    kind = _messages.StringField(2, default='dns#rRSetRoutingPolicyPrimaryBackupPolicy')
    primaryTargets = _messages.MessageField('RRSetRoutingPolicyHealthCheckTargets', 3)
    trickleTraffic = _messages.FloatField(4)