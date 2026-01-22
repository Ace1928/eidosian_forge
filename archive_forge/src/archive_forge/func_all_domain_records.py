from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.digitalocean.plugins.module_utils.digital_ocean import (
import time
def all_domain_records(self):
    resp = self.get('domains/%s/records/' % self.domain_name)
    return resp.json