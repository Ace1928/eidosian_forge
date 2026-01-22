from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyAdaptiveProtectionConfigLayer7DdosDefenseConfigThresholdConfigTrafficGranularityConfig(_messages.Message):
    """Configurations to specifc granular traffic units processed by Adaptive
  Protection.

  Enums:
    TypeValueValuesEnum: Type of this configuration.

  Fields:
    enableEachUniqueValue: If enabled, traffic matching each unique value for
      the specified type constitutes a separate traffic unit. It can only be
      set to true if `value` is empty.
    type: Type of this configuration.
    value: Requests that match this value constitute a granular traffic unit.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of this configuration.

    Values:
      HTTP_HEADER_HOST: <no description>
      HTTP_PATH: <no description>
      UNSPECIFIED_TYPE: <no description>
    """
        HTTP_HEADER_HOST = 0
        HTTP_PATH = 1
        UNSPECIFIED_TYPE = 2
    enableEachUniqueValue = _messages.BooleanField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)
    value = _messages.StringField(3)