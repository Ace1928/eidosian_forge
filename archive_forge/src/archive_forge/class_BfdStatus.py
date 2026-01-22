from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BfdStatus(_messages.Message):
    """Next free: 15

  Enums:
    BfdSessionInitializationModeValueValuesEnum: The BFD session
      initialization mode for this BGP peer. If set to ACTIVE, the Cloud
      Router will initiate the BFD session for this BGP peer. If set to
      PASSIVE, the Cloud Router will wait for the peer router to initiate the
      BFD session for this BGP peer. If set to DISABLED, BFD is disabled for
      this BGP peer.
    LocalDiagnosticValueValuesEnum: The diagnostic code specifies the local
      system's reason for the last change in session state. This allows remote
      systems to determine the reason that the previous session failed, for
      example. These diagnostic codes are specified in section 4.1 of RFC5880
    LocalStateValueValuesEnum: The current BFD session state as seen by the
      transmitting system. These states are specified in section 4.1 of
      RFC5880

  Fields:
    bfdSessionInitializationMode: The BFD session initialization mode for this
      BGP peer. If set to ACTIVE, the Cloud Router will initiate the BFD
      session for this BGP peer. If set to PASSIVE, the Cloud Router will wait
      for the peer router to initiate the BFD session for this BGP peer. If
      set to DISABLED, BFD is disabled for this BGP peer.
    configUpdateTimestampMicros: Unix timestamp of the most recent config
      update.
    controlPacketCounts: Control packet counts for the current BFD session.
    controlPacketIntervals: Inter-packet time interval statistics for control
      packets.
    localDiagnostic: The diagnostic code specifies the local system's reason
      for the last change in session state. This allows remote systems to
      determine the reason that the previous session failed, for example.
      These diagnostic codes are specified in section 4.1 of RFC5880
    localState: The current BFD session state as seen by the transmitting
      system. These states are specified in section 4.1 of RFC5880
    negotiatedLocalControlTxIntervalMs: Negotiated transmit interval for
      control packets.
    rxPacket: The most recent Rx control packet for this BFD session.
    txPacket: The most recent Tx control packet for this BFD session.
    uptimeMs: Session uptime in milliseconds. Value will be 0 if session is
      not up.
  """

    class BfdSessionInitializationModeValueValuesEnum(_messages.Enum):
        """The BFD session initialization mode for this BGP peer. If set to
    ACTIVE, the Cloud Router will initiate the BFD session for this BGP peer.
    If set to PASSIVE, the Cloud Router will wait for the peer router to
    initiate the BFD session for this BGP peer. If set to DISABLED, BFD is
    disabled for this BGP peer.

    Values:
      ACTIVE: <no description>
      DISABLED: <no description>
      PASSIVE: <no description>
    """
        ACTIVE = 0
        DISABLED = 1
        PASSIVE = 2

    class LocalDiagnosticValueValuesEnum(_messages.Enum):
        """The diagnostic code specifies the local system's reason for the last
    change in session state. This allows remote systems to determine the
    reason that the previous session failed, for example. These diagnostic
    codes are specified in section 4.1 of RFC5880

    Values:
      ADMINISTRATIVELY_DOWN: <no description>
      CONCATENATED_PATH_DOWN: <no description>
      CONTROL_DETECTION_TIME_EXPIRED: <no description>
      DIAGNOSTIC_UNSPECIFIED: <no description>
      ECHO_FUNCTION_FAILED: <no description>
      FORWARDING_PLANE_RESET: <no description>
      NEIGHBOR_SIGNALED_SESSION_DOWN: <no description>
      NO_DIAGNOSTIC: <no description>
      PATH_DOWN: <no description>
      REVERSE_CONCATENATED_PATH_DOWN: <no description>
    """
        ADMINISTRATIVELY_DOWN = 0
        CONCATENATED_PATH_DOWN = 1
        CONTROL_DETECTION_TIME_EXPIRED = 2
        DIAGNOSTIC_UNSPECIFIED = 3
        ECHO_FUNCTION_FAILED = 4
        FORWARDING_PLANE_RESET = 5
        NEIGHBOR_SIGNALED_SESSION_DOWN = 6
        NO_DIAGNOSTIC = 7
        PATH_DOWN = 8
        REVERSE_CONCATENATED_PATH_DOWN = 9

    class LocalStateValueValuesEnum(_messages.Enum):
        """The current BFD session state as seen by the transmitting system.
    These states are specified in section 4.1 of RFC5880

    Values:
      ADMIN_DOWN: <no description>
      DOWN: <no description>
      INIT: <no description>
      STATE_UNSPECIFIED: <no description>
      UP: <no description>
    """
        ADMIN_DOWN = 0
        DOWN = 1
        INIT = 2
        STATE_UNSPECIFIED = 3
        UP = 4
    bfdSessionInitializationMode = _messages.EnumField('BfdSessionInitializationModeValueValuesEnum', 1)
    configUpdateTimestampMicros = _messages.IntegerField(2)
    controlPacketCounts = _messages.MessageField('BfdStatusPacketCounts', 3)
    controlPacketIntervals = _messages.MessageField('PacketIntervals', 4, repeated=True)
    localDiagnostic = _messages.EnumField('LocalDiagnosticValueValuesEnum', 5)
    localState = _messages.EnumField('LocalStateValueValuesEnum', 6)
    negotiatedLocalControlTxIntervalMs = _messages.IntegerField(7, variant=_messages.Variant.UINT32)
    rxPacket = _messages.MessageField('BfdPacket', 8)
    txPacket = _messages.MessageField('BfdPacket', 9)
    uptimeMs = _messages.IntegerField(10)