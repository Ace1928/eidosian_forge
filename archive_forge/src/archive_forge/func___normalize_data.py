from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
def __normalize_data(self):
    if self.payload['type'] in ['CNAME', 'MX', 'SRV', 'CAA'] and self.payload['data'] != '@' and (not self.payload['data'].endswith('.')):
        data = '%s.' % self.payload['data']
    else:
        data = self.payload['data']
    return data