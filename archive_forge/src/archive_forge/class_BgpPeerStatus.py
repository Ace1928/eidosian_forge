from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BgpPeerStatus(_messages.Message):
    """Status of a BGP peer.

  Enums:
    StatusValueValuesEnum: The current status of BGP.

  Fields:
    ipAddress: IP address of the local BGP interface.
    name: Name of this BGP peer. Unique within the Routers resource.
    peerIpAddress: IP address of the remote BGP interface.
    prefixCounter: A collection of counts for prefixes.
    state: BGP state as specified in RFC1771.
    status: The current status of BGP.
    uptime: Time this session has been up. Format: 14 years, 51 weeks, 6 days,
      23 hours, 59 minutes, 59 seconds
    uptimeSeconds: Time this session has been up, in seconds.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """The current status of BGP.

    Values:
      UNKNOWN: The default status indicating BGP session is in unknown state.
      UP: The UP status indicating BGP session is established.
      DOWN: The DOWN state indicating BGP session is not established yet.
    """
        UNKNOWN = 0
        UP = 1
        DOWN = 2
    ipAddress = _messages.StringField(1)
    name = _messages.StringField(2)
    peerIpAddress = _messages.StringField(3)
    prefixCounter = _messages.MessageField('PrefixCounter', 4)
    state = _messages.StringField(5)
    status = _messages.EnumField('StatusValueValuesEnum', 6)
    uptime = _messages.StringField(7)
    uptimeSeconds = _messages.IntegerField(8)