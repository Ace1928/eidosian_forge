from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocationMetadata(_messages.Message):
    """VmwareEngine specific metadata for the given
  google.cloud.location.Location. It is returned as a content of the
  `google.cloud.location.Location.metadata` field.

  Enums:
    CapabilitiesValueListEntryValuesEnum:

  Fields:
    capabilities: Output only. Capabilities of this location.
  """

    class CapabilitiesValueListEntryValuesEnum(_messages.Enum):
        """CapabilitiesValueListEntryValuesEnum enum type.

    Values:
      CAPABILITY_UNSPECIFIED: The default value. This value is used if the
        capability is omitted or unknown.
      STRETCHED_CLUSTERS: Stretch clusters are supported in this location.
    """
        CAPABILITY_UNSPECIFIED = 0
        STRETCHED_CLUSTERS = 1
    capabilities = _messages.EnumField('CapabilitiesValueListEntryValuesEnum', 1, repeated=True)