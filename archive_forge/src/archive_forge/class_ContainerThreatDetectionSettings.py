from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerThreatDetectionSettings(_messages.Message):
    """Resource capturing the settings for the Container Threat Detection
  service.

  Enums:
    ServiceEnablementStateValueValuesEnum: The state of enablement for the
      service at its level of the resource hierarchy. A DISABLED state will
      override all module enablement_states to DISABLED.

  Messages:
    ModulesValue: The configurations including the state of enablement for the
      service's different modules. The absence of a module in the map implies
      its configuration is inherited from its parent's.

  Fields:
    modules: The configurations including the state of enablement for the
      service's different modules. The absence of a module in the map implies
      its configuration is inherited from its parent's.
    name: The resource name of the ContainerThreatDetectionSettings. Formats:
      * organizations/{organization}/containerThreatDetectionSettings *
      folders/{folder}/containerThreatDetectionSettings *
      projects/{project}/containerThreatDetectionSettings * projects/{project}
      /locations/{location}/clusters/{cluster}/containerThreatDetectionSetting
      s
    serviceAccount: Output only. The service account used by Container Threat
      Detection for scanning. Service accounts are scoped at the project level
      meaning this field will be empty at any level above a project.
    serviceEnablementState: The state of enablement for the service at its
      level of the resource hierarchy. A DISABLED state will override all
      module enablement_states to DISABLED.
    updateTime: Output only. The time the settings were last updated.
  """

    class ServiceEnablementStateValueValuesEnum(_messages.Enum):
        """The state of enablement for the service at its level of the resource
    hierarchy. A DISABLED state will override all module enablement_states to
    DISABLED.

    Values:
      ENABLEMENT_STATE_UNSPECIFIED: Default value. This value is unused.
      INHERITED: State is inherited from the parent resource.
      ENABLED: State is enabled.
      DISABLED: State is disabled.
    """
        ENABLEMENT_STATE_UNSPECIFIED = 0
        INHERITED = 1
        ENABLED = 2
        DISABLED = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ModulesValue(_messages.Message):
        """The configurations including the state of enablement for the service's
    different modules. The absence of a module in the map implies its
    configuration is inherited from its parent's.

    Messages:
      AdditionalProperty: An additional property for a ModulesValue object.

    Fields:
      additionalProperties: Additional properties of type ModulesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ModulesValue object.

      Fields:
        key: Name of the additional property.
        value: A Config attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('Config', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    modules = _messages.MessageField('ModulesValue', 1)
    name = _messages.StringField(2)
    serviceAccount = _messages.StringField(3)
    serviceEnablementState = _messages.EnumField('ServiceEnablementStateValueValuesEnum', 4)
    updateTime = _messages.StringField(5)