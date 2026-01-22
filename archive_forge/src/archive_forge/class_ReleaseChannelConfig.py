from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReleaseChannelConfig(_messages.Message):
    """ReleaseChannelConfig exposes configuration for a release channel.

  Enums:
    ChannelValueValuesEnum: The release channel this configuration applies to.

  Fields:
    channel: The release channel this configuration applies to.
    defaultVersion: The default version for newly created clusters on the
      channel.
    validVersions: List of valid versions for the channel.
  """

    class ChannelValueValuesEnum(_messages.Enum):
        """The release channel this configuration applies to.

    Values:
      UNSPECIFIED: No channel specified.
      RAPID: RAPID channel is offered on an early access basis for customers
        who want to test new releases. WARNING: Versions available in the
        RAPID Channel may be subject to unresolved issues with no known
        workaround and are not subject to any SLAs.
      REGULAR: Clusters subscribed to REGULAR receive versions that are
        considered GA quality. REGULAR is intended for production users who
        want to take advantage of new features.
      STABLE: Clusters subscribed to STABLE receive versions that are known to
        be stable and reliable in production.
    """
        UNSPECIFIED = 0
        RAPID = 1
        REGULAR = 2
        STABLE = 3
    channel = _messages.EnumField('ChannelValueValuesEnum', 1)
    defaultVersion = _messages.StringField(2)
    validVersions = _messages.StringField(3, repeated=True)