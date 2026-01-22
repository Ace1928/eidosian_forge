from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def __get_registry_fw_less_than_3(self):
    reg = {}
    network_device_function_id = self.module.params.get('network_device_function_id')
    scp_response = self.idrac.export_scp(export_format='JSON', export_use='Default', target='NIC', job_wait=True)
    comp = scp_response.json_data.get('SystemConfiguration', {}).get('Components', {})
    for each in comp:
        if each.get('FQDD') == network_device_function_id:
            for each_attr in each.get('Attributes'):
                reg.update({each_attr['Name']: each_attr['Value']})
    return reg