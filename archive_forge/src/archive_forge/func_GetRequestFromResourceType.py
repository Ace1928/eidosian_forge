from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import service as settings_service
def GetRequestFromResourceType(resource_type, setting_name, is_effective):
    """Returns the get_request from the user-specified arguments.

  Args:
    resource_type: A String object that contains the resource type
    setting_name: setting name such as `settings/iam-projectCreatorRoles`
    is_effective: indicate if it is requesting for an effective setting
  """
    messages = settings_service.ResourceSettingsMessages()
    if resource_type == ORGANIZATION:
        view = messages.ResourcesettingsOrganizationsSettingsGetRequest.ViewValueValuesEnum.SETTING_VIEW_EFFECTIVE_VALUE if is_effective else messages.ResourcesettingsOrganizationsSettingsGetRequest.ViewValueValuesEnum.SETTING_VIEW_LOCAL_VALUE
        get_request = messages.ResourcesettingsOrganizationsSettingsGetRequest(name=setting_name, view=view)
    elif resource_type == FOLDER:
        view = messages.ResourcesettingsFoldersSettingsGetRequest.ViewValueValuesEnum.SETTING_VIEW_EFFECTIVE_VALUE if is_effective else messages.ResourcesettingsFoldersSettingsGetRequest.ViewValueValuesEnum.SETTING_VIEW_LOCAL_VALUE
        get_request = messages.ResourcesettingsFoldersSettingsGetRequest(name=setting_name, view=view)
    else:
        view = messages.ResourcesettingsProjectsSettingsGetRequest.ViewValueValuesEnum.SETTING_VIEW_EFFECTIVE_VALUE if is_effective else messages.ResourcesettingsProjectsSettingsGetRequest.ViewValueValuesEnum.SETTING_VIEW_LOCAL_VALUE
        get_request = messages.ResourcesettingsProjectsSettingsGetRequest(name=setting_name, view=view)
    return get_request