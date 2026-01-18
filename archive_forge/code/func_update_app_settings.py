from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_app_settings(self):
    """
        Update application settings
        :return: deserialized updating response
        """
    self.log('Update application setting')
    try:
        settings = StringDictionary(properties=self.app_settings_strDic)
        response = self.web_client.web_apps.update_application_settings(resource_group_name=self.resource_group, name=self.name, app_settings=settings)
        self.log('Response : {0}'.format(response))
        return response
    except Exception as ex:
        self.fail('Failed to update application settings for web app {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))