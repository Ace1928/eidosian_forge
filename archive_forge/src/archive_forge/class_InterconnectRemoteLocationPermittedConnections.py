from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InterconnectRemoteLocationPermittedConnections(_messages.Message):
    """A InterconnectRemoteLocationPermittedConnections object.

  Fields:
    interconnectLocation: [Output Only] URL of an Interconnect location that
      is permitted to connect to this Interconnect remote location.
  """
    interconnectLocation = _messages.StringField(1)