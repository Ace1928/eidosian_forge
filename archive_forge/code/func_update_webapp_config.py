from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def update_webapp_config(self, param):
    try:
        return self.web_client.web_apps.create_or_update_configuration(resource_group_name=self.resource_group, name=self.name, site_config=param)
    except Exception as exc:
        self.fail('Error creating/updating webapp config {0} (rg={1}) - {2}'.format(self.name, self.resource_group, str(exc)))