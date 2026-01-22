from __future__ import (absolute_import, division, print_function)
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
def _get_all_devices(self, device_uri):
    device_host = []
    device_host_uri = device_uri.strip('/api/')
    port = self.get_option('port') if 'port' in self.config else 443
    validate_certs = self.get_option('validate_certs') if 'validate_certs' in self.config else False
    module_params = {'hostname': self.get_option('hostname'), 'username': self.get_option('username'), 'password': self.get_option('password'), 'port': port, 'validate_certs': validate_certs}
    if 'ca_path' in self.config:
        module_params.update({'ca_path': self.get_option('ca_path')})
    with RestOME(module_params, req_session=False) as ome:
        device_resp = get_all_data_with_pagination(ome, device_host_uri)
        device_data = device_resp.get('report_list', [])
        if device_data is not None:
            for mgmt in device_data:
                if len(mgmt['DeviceManagement']) != 0:
                    device_host.append(self._get_device_host(mgmt))
    return device_host