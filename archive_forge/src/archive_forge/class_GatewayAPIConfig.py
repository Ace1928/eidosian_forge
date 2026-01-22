from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GatewayAPIConfig(_messages.Message):
    """GatewayAPIConfig contains the desired config of Gateway API on this
  cluster.

  Enums:
    ChannelValueValuesEnum: The Gateway API release channel to use for Gateway
      API.

  Fields:
    channel: The Gateway API release channel to use for Gateway API.
  """

    class ChannelValueValuesEnum(_messages.Enum):
        """The Gateway API release channel to use for Gateway API.

    Values:
      CHANNEL_UNSPECIFIED: Default value.
      CHANNEL_DISABLED: Gateway API support is disabled
      CHANNEL_EXPERIMENTAL: Gateway API support is enabled, experimental CRDs
        are installed
      CHANNEL_STANDARD: Gateway API support is enabled, standard CRDs are
        installed
    """
        CHANNEL_UNSPECIFIED = 0
        CHANNEL_DISABLED = 1
        CHANNEL_EXPERIMENTAL = 2
        CHANNEL_STANDARD = 3
    channel = _messages.EnumField('ChannelValueValuesEnum', 1)