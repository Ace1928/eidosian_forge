from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_app_settings_slot(self, slot_name=None, app_settings=None):
    """
        Update application settings
        :return: deserialized updating response
        """
    self.log('Update application setting')
    if slot_name is None:
        slot_name = self.name
    if app_settings is None:
        app_settings = self.app_settings_strDic
    try:
        settings = StringDictionary(properties=self.app_settings)
        response = self.web_client.web_apps.update_application_settings_slot(resource_group_name=self.resource_group, name=self.webapp_name, slot=slot_name, app_settings=settings)
        self.log('Response : {0}'.format(response))
        return response.as_dict()
    except Exception as ex:
        self.fail('Failed to update application settings for web app slot {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
    return response