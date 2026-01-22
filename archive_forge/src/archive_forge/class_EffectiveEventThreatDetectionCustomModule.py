from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectiveEventThreatDetectionCustomModule(_messages.Message):
    """An EffectiveEventThreatDetectionCustomModule is the representation of an
  Event Threat Detection custom module at a specified level of the resource
  hierarchy: organization, folder, or project. If a custom module is inherited
  from a parent organization or folder, the value of the `enablement_state`
  property in EffectiveEventThreatDetectionCustomModule is set to the value
  that is effective in the parent, instead of `INHERITED`. For example, if the
  module is enabled in a parent organization or folder, the effective
  `enablement_state` for the module in all child folders or projects is also
  `enabled`. EffectiveEventThreatDetectionCustomModule is read-only.

  Enums:
    EnablementStateValueValuesEnum: Output only. The effective state of
      enablement for the module at the given level of the hierarchy.

  Messages:
    ConfigValue: Output only. Config for the effective module.

  Fields:
    config: Output only. Config for the effective module.
    description: Output only. The description for the module.
    displayName: Output only. The human readable name to be displayed for the
      module.
    enablementState: Output only. The effective state of enablement for the
      module at the given level of the hierarchy.
    name: Output only. The resource name of the effective ETD custom module.
      Its format is: * "organizations/{organization}/eventThreatDetectionSetti
      ngs/effectiveCustomModules/{module}". * "folders/{folder}/eventThreatDet
      ectionSettings/effectiveCustomModules/{module}". * "projects/{project}/e
      ventThreatDetectionSettings/effectiveCustomModules/{module}".
    type: Output only. Type for the module. e.g. CONFIGURABLE_BAD_IP.
  """

    class EnablementStateValueValuesEnum(_messages.Enum):
        """Output only. The effective state of enablement for the module at the
    given level of the hierarchy.

    Values:
      ENABLEMENT_STATE_UNSPECIFIED: Unspecified enablement state.
      ENABLED: The module is enabled at the given level.
      DISABLED: The module is disabled at the given level.
    """
        ENABLEMENT_STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConfigValue(_messages.Message):
        """Output only. Config for the effective module.

    Messages:
      AdditionalProperty: An additional property for a ConfigValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConfigValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    config = _messages.MessageField('ConfigValue', 1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    enablementState = _messages.EnumField('EnablementStateValueValuesEnum', 4)
    name = _messages.StringField(5)
    type = _messages.StringField(6)