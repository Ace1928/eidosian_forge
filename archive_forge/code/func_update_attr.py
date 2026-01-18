from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, env_fallback
def update_attr(self, attrs=None):
    if attrs:
        for k, v in attrs.items():
            setattr(self, k, v)
        networks = attrs.get('networks', {})
        for network in networks.get('v6', []):
            if network['type'] == 'public':
                setattr(self, 'public_ipv6_address', network['ip_address'])
            else:
                setattr(self, 'private_ipv6_address', network['ip_address'])
    else:
        json = self.manager.show_droplet(self.id)
        if json['ip_address']:
            self.update_attr(json)