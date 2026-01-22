from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _needs_to_change_profile_config(self, key):
    if key not in self.config:
        return False
    old_configs = self.old_profile_json['metadata'].get(key, None)
    return self.config[key] != old_configs