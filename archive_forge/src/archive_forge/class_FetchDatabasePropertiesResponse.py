from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FetchDatabasePropertiesResponse(_messages.Message):
    """Response for FetchDatabasePropertiesRequest.

  Fields:
    isFailoverReplicaAvailable: The availability status of the failover
      replica. A false status indicates that the failover replica is out of
      sync. The primary instance can only fail over to the failover replica
      when the status is true.
    primaryGceZone: The Compute Engine zone that the instance is currently
      serving from.
    secondaryGceZone: The Compute Engine zone that the failover instance is
      currently serving from for a regional Cloud SQL instance.
  """
    isFailoverReplicaAvailable = _messages.BooleanField(1)
    primaryGceZone = _messages.StringField(2)
    secondaryGceZone = _messages.StringField(3)