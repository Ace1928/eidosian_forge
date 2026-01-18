from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
import re
def update_instance_tags(self, tags):
    try:
        poller = self.IoThub_client.iot_hub_resource.begin_update(self.resource_group, self.name, tags=tags)
        return self.get_poller_result(poller)
    except Exception as exc:
        self.fail("Error updating IoT Hub {0}'s tag: {1}".format(self.name, exc.message or str(exc)))