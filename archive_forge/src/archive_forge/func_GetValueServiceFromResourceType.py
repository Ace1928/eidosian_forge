from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import service as settings_service
def GetValueServiceFromResourceType(resource_type):
    """Returns the value-service from the resource type input.

  Args:
    resource_type: A String object that contains the resource type
  """
    if resource_type == ORGANIZATION:
        value_service = settings_service.OrganizationsSettingsValueService()
    elif resource_type == FOLDER:
        value_service = settings_service.FoldersSettingsValueService()
    else:
        value_service = settings_service.ProjectsSettingsValueService()
    return value_service