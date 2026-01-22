from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventThreatDetectionCustomModule(_messages.Message):
    """Represents an instance of an Event Threat Detection custom module,
  including its full module name, display name, enablement state, and last
  updated time. You can create a custom module at the organization, folder, or
  project level. Custom modules that you create at the organization or folder
  level are inherited by child folders and projects.

  Enums:
    EnablementStateValueValuesEnum: The state of enablement for the module at
      the given level of the hierarchy.

  Messages:
    ConfigValue: Config for the module. For the resident module, its config
      value is defined at this level. For the inherited module, its config
      value is inherited from the ancestor module.

  Fields:
    ancestorModule: Output only. The closest ancestor module that this module
      inherits the enablement state from. The format is the same as the
      EventThreatDetectionCustomModule resource name.
    config: Config for the module. For the resident module, its config value
      is defined at this level. For the inherited module, its config value is
      inherited from the ancestor module.
    description: The description for the module.
    displayName: The human readable name to be displayed for the module.
    enablementState: The state of enablement for the module at the given level
      of the hierarchy.
    lastEditor: Output only. The editor the module was last updated by.
    name: Immutable. The resource name of the Event Threat Detection custom
      module. Its format is: * "organizations/{organization}/eventThreatDetect
      ionSettings/customModules/{module}". *
      "folders/{folder}/eventThreatDetectionSettings/customModules/{module}".
      * "projects/{project}/eventThreatDetectionSettings/customModules/{module
      }".
    type: Type for the module. e.g. CONFIGURABLE_BAD_IP.
    updateTime: Output only. The time the module was last updated.
  """

    class EnablementStateValueValuesEnum(_messages.Enum):
        """The state of enablement for the module at the given level of the
    hierarchy.

    Values:
      ENABLEMENT_STATE_UNSPECIFIED: Unspecified enablement state.
      ENABLED: The module is enabled at the given level.
      DISABLED: The module is disabled at the given level.
      INHERITED: When the enablement state is inherited.
    """
        ENABLEMENT_STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
        INHERITED = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConfigValue(_messages.Message):
        """Config for the module. For the resident module, its config value is
    defined at this level. For the inherited module, its config value is
    inherited from the ancestor module.

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
    ancestorModule = _messages.StringField(1)
    config = _messages.MessageField('ConfigValue', 2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    enablementState = _messages.EnumField('EnablementStateValueValuesEnum', 5)
    lastEditor = _messages.StringField(6)
    name = _messages.StringField(7)
    type = _messages.StringField(8)
    updateTime = _messages.StringField(9)