from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectiveSecurityHealthAnalyticsCustomModule(_messages.Message):
    """An EffectiveSecurityHealthAnalyticsCustomModule is the representation of
  a Security Health Analytics custom module at a specified level of the
  resource hierarchy: organization, folder, or project. If a custom module is
  inherited from a parent organization or folder, the value of the
  `enablementState` property in EffectiveSecurityHealthAnalyticsCustomModule
  is set to the value that is effective in the parent, instead of `INHERITED`.
  For example, if the module is enabled in a parent organization or folder,
  the effective enablement_state for the module in all child folders or
  projects is also `enabled`. EffectiveSecurityHealthAnalyticsCustomModule is
  read-only.

  Enums:
    EnablementStateValueValuesEnum: Output only. The effective state of
      enablement for the module at the given level of the hierarchy.

  Fields:
    customConfig: Output only. The user-specified configuration for the
      module.
    displayName: Output only. The display name for the custom module. The name
      must be between 1 and 128 characters, start with a lowercase letter, and
      contain alphanumeric characters or underscores only.
    enablementState: Output only. The effective state of enablement for the
      module at the given level of the hierarchy.
    name: Identifier. The resource name of the custom module. Its format is "o
      rganizations/{organization}/locations/{location}/effectiveSecurityHealth
      AnalyticsCustomModules/{effective_security_health_analytics_custom_modul
      e}", or "folders/{folder}/locations/{location}/effectiveSecurityHealthAn
      alyticsCustomModules/{effective_security_health_analytics_custom_module}
      ", or "projects/{project}/locations/{location}/effectiveSecurityHealthAn
      alyticsCustomModules/{effective_security_health_analytics_custom_module}
      "
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
    customConfig = _messages.MessageField('CustomConfig', 1)
    displayName = _messages.StringField(2)
    enablementState = _messages.EnumField('EnablementStateValueValuesEnum', 3)
    name = _messages.StringField(4)