from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.facts import ansible_collector, default_collectors
def _get_api_ip(self):
    """Return the IP of the DHCP server."""
    if module.params.get('meta_data_host'):
        return module.params.get('meta_data_host')
    elif not self.api_ip:
        dhcp_lease_file = self._get_dhcp_lease_file()
        for line in open(dhcp_lease_file):
            if 'dhcp-server-identifier' in line:
                line = line.translate(None, ';')
                self.api_ip = line.split()[2]
                break
        if not self.api_ip:
            module.fail_json(msg='No dhcp-server-identifier found in leases file.')
    return self.api_ip