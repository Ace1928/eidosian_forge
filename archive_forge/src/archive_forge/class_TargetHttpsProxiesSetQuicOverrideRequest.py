from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetHttpsProxiesSetQuicOverrideRequest(_messages.Message):
    """A TargetHttpsProxiesSetQuicOverrideRequest object.

  Enums:
    QuicOverrideValueValuesEnum: QUIC policy for the TargetHttpsProxy
      resource.

  Fields:
    quicOverride: QUIC policy for the TargetHttpsProxy resource.
  """

    class QuicOverrideValueValuesEnum(_messages.Enum):
        """QUIC policy for the TargetHttpsProxy resource.

    Values:
      DISABLE: The load balancer will not attempt to negotiate QUIC with
        clients.
      ENABLE: The load balancer will attempt to negotiate QUIC with clients.
      NONE: No overrides to the default QUIC policy. This option is implicit
        if no QUIC override has been specified in the request.
    """
        DISABLE = 0
        ENABLE = 1
        NONE = 2
    quicOverride = _messages.EnumField('QuicOverrideValueValuesEnum', 1)