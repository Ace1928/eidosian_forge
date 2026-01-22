from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1SecurityHealthAnalyticsCustomModule(_messages.Message):
    """Represents an instance of a Security Health Analytics custom module,
  including its full module name, display name, enablement state, and last
  updated time. You can create a custom module at the organization, folder, or
  project level. Custom modules that you create at the organization or folder
  level are inherited by the child folders and projects.

  Enums:
    EnablementStateValueValuesEnum: The enablement state of the custom module.

  Fields:
    ancestorModule: Output only. If empty, indicates that the custom module
      was created in the organization, folder, or project in which you are
      viewing the custom module. Otherwise, `ancestor_module` specifies the
      organization or folder from which the custom module is inherited.
    customConfig: The user specified custom configuration for the module.
    displayName: The display name of the Security Health Analytics custom
      module. This display name becomes the finding category for all findings
      that are returned by this custom module. The display name must be
      between 1 and 128 characters, start with a lowercase letter, and contain
      alphanumeric characters or underscores only.
    enablementState: The enablement state of the custom module.
    lastEditor: Output only. The editor that last updated the custom module.
    name: Immutable. The resource name of the custom module. Its format is "or
      ganizations/{organization}/securityHealthAnalyticsSettings/customModules
      /{customModule}", or "folders/{folder}/securityHealthAnalyticsSettings/c
      ustomModules/{customModule}", or "projects/{project}/securityHealthAnaly
      ticsSettings/customModules/{customModule}" The id {customModule} is
      server-generated and is not user settable. It will be a numeric id
      containing 1-20 digits.
    updateTime: Output only. The time at which the custom module was last
      updated.
  """

    class EnablementStateValueValuesEnum(_messages.Enum):
        """The enablement state of the custom module.

    Values:
      ENABLEMENT_STATE_UNSPECIFIED: Unspecified enablement state.
      ENABLED: The module is enabled at the given CRM resource.
      DISABLED: The module is disabled at the given CRM resource.
      INHERITED: State is inherited from an ancestor module. The module will
        either be effectively ENABLED or DISABLED based on its closest non-
        inherited ancestor module in the CRM hierarchy.
    """
        ENABLEMENT_STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
        INHERITED = 3
    ancestorModule = _messages.StringField(1)
    customConfig = _messages.MessageField('GoogleCloudSecuritycenterV1CustomConfig', 2)
    displayName = _messages.StringField(3)
    enablementState = _messages.EnumField('EnablementStateValueValuesEnum', 4)
    lastEditor = _messages.StringField(5)
    name = _messages.StringField(6)
    updateTime = _messages.StringField(7)