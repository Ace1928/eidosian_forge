from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ManagedZoneServiceDirectoryConfig(_messages.Message):
    """Contains information about Service Directory-backed zones.

  Fields:
    kind: A string attribute.
    namespace: Contains information about the namespace associated with the
      zone.
  """
    kind = _messages.StringField(1, default='dns#managedZoneServiceDirectoryConfig')
    namespace = _messages.MessageField('ManagedZoneServiceDirectoryConfigNamespace', 2)