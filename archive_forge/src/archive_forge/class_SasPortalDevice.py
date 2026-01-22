from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalDevice(_messages.Message):
    """A SasPortalDevice object.

  Enums:
    StateValueValuesEnum: Output only. Device state.

  Fields:
    activeConfig: Output only. Current configuration of the device as
      registered to the SAS.
    currentChannels: Output only. Current channels with scores.
    deviceMetadata: Device parameters that can be overridden by both SAS
      Portal and SAS registration requests.
    displayName: Device display name.
    fccId: The FCC identifier of the device. Refer to
      https://www.fcc.gov/oet/ea/fccid for FccID format. Accept underscores
      and periods because some test-SAS customers use them.
    grantRangeAllowlists: Only ranges that are within the allowlists are
      available for new grants.
    grants: Output only. Grants held by the device.
    name: Output only. The resource path name.
    preloadedConfig: Configuration of the device, as specified via SAS Portal
      API.
    serialNumber: A serial number assigned to the device by the device
      manufacturer.
    state: Output only. Device state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Device state.

    Values:
      DEVICE_STATE_UNSPECIFIED: Unspecified state.
      RESERVED: Device created in the SAS Portal, however, not yet registered
        with SAS.
      REGISTERED: Device registered with SAS.
      DEREGISTERED: Device de-registered with SAS.
    """
        DEVICE_STATE_UNSPECIFIED = 0
        RESERVED = 1
        REGISTERED = 2
        DEREGISTERED = 3
    activeConfig = _messages.MessageField('SasPortalDeviceConfig', 1)
    currentChannels = _messages.MessageField('SasPortalChannelWithScore', 2, repeated=True)
    deviceMetadata = _messages.MessageField('SasPortalDeviceMetadata', 3)
    displayName = _messages.StringField(4)
    fccId = _messages.StringField(5)
    grantRangeAllowlists = _messages.MessageField('SasPortalFrequencyRange', 6, repeated=True)
    grants = _messages.MessageField('SasPortalDeviceGrant', 7, repeated=True)
    name = _messages.StringField(8)
    preloadedConfig = _messages.MessageField('SasPortalDeviceConfig', 9)
    serialNumber = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)