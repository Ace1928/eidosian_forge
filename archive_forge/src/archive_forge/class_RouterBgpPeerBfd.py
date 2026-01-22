from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouterBgpPeerBfd(_messages.Message):
    """A RouterBgpPeerBfd object.

  Enums:
    SessionInitializationModeValueValuesEnum: The BFD session initialization
      mode for this BGP peer. If set to ACTIVE, the Cloud Router will initiate
      the BFD session for this BGP peer. If set to PASSIVE, the Cloud Router
      will wait for the peer router to initiate the BFD session for this BGP
      peer. If set to DISABLED, BFD is disabled for this BGP peer. The default
      is DISABLED.

  Fields:
    minReceiveInterval: The minimum interval, in milliseconds, between BFD
      control packets received from the peer router. The actual value is
      negotiated between the two routers and is equal to the greater of this
      value and the transmit interval of the other router. If set, this value
      must be between 1000 and 30000. The default is 1000.
    minTransmitInterval: The minimum interval, in milliseconds, between BFD
      control packets transmitted to the peer router. The actual value is
      negotiated between the two routers and is equal to the greater of this
      value and the corresponding receive interval of the other router. If
      set, this value must be between 1000 and 30000. The default is 1000.
    multiplier: The number of consecutive BFD packets that must be missed
      before BFD declares that a peer is unavailable. If set, the value must
      be a value between 5 and 16. The default is 5.
    sessionInitializationMode: The BFD session initialization mode for this
      BGP peer. If set to ACTIVE, the Cloud Router will initiate the BFD
      session for this BGP peer. If set to PASSIVE, the Cloud Router will wait
      for the peer router to initiate the BFD session for this BGP peer. If
      set to DISABLED, BFD is disabled for this BGP peer. The default is
      DISABLED.
  """

    class SessionInitializationModeValueValuesEnum(_messages.Enum):
        """The BFD session initialization mode for this BGP peer. If set to
    ACTIVE, the Cloud Router will initiate the BFD session for this BGP peer.
    If set to PASSIVE, the Cloud Router will wait for the peer router to
    initiate the BFD session for this BGP peer. If set to DISABLED, BFD is
    disabled for this BGP peer. The default is DISABLED.

    Values:
      ACTIVE: <no description>
      DISABLED: <no description>
      PASSIVE: <no description>
    """
        ACTIVE = 0
        DISABLED = 1
        PASSIVE = 2
    minReceiveInterval = _messages.IntegerField(1, variant=_messages.Variant.UINT32)
    minTransmitInterval = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    multiplier = _messages.IntegerField(3, variant=_messages.Variant.UINT32)
    sessionInitializationMode = _messages.EnumField('SessionInitializationModeValueValuesEnum', 4)