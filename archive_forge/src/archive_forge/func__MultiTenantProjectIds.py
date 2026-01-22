from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.container import api_adapter as container_api_adapter
from googlecloudsdk.api_lib.container.fleet import client as hub_client
from googlecloudsdk.api_lib.container.fleet import util as hub_util
from googlecloudsdk.api_lib.resourcesettings import service as resourcesettings_service
from googlecloudsdk.api_lib.run import job
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.runtime_config import util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _MultiTenantProjectIds(project):
    """Returns a list of Multitenant project ids."""
    setting_name = 'projects/{}/settings/cloudrun-multiTenancy'.format(project)
    messages = resourcesettings_service.ResourceSettingsMessages()
    get_request = messages.ResourcesettingsProjectsSettingsGetRequest(name=setting_name, view=messages.ResourcesettingsProjectsSettingsGetRequest.ViewValueValuesEnum.SETTING_VIEW_EFFECTIVE_VALUE)
    settings_service = resourcesettings_service.ProjectsSettingsService()
    service_value = settings_service.LookupEffectiveValue(get_request)
    return [_MulitTenantProjectId(project) for project in service_value.localValue.stringSetValue.values]