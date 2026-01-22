from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def __get_registry_fw_less_than_6_more_than_3(self):
    reg = {}
    network_device_function_id = self.module.params.get('network_device_function_id')
    registry = get_dynamic_uri(self.idrac, REGISTRY_URI, 'Members')
    for each_member in registry:
        if network_device_function_id in each_member.get('@odata.id'):
            location = get_dynamic_uri(self.idrac, each_member.get('@odata.id'), 'Location')
            if location:
                uri = location[0].get('Uri')
                attr = get_dynamic_uri(self.idrac, uri, 'RegistryEntries').get('Attributes', {})
                for each_attr in attr:
                    reg.update({each_attr['AttributeName']: each_attr['CurrentValue']})
                break
    return reg