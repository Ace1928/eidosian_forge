from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZoneCloudLoggingConfig(_messages.Message):
    """Cloud Logging configurations for publicly visible zones.

  Fields:
    enableLogging: If set, enable query logging for this ManagedZone. False by
      default, making logging opt-in.
    kind: A string attribute.
  """
    enableLogging = _messages.BooleanField(1)
    kind = _messages.StringField(2, default='dns#managedZoneCloudLoggingConfig')