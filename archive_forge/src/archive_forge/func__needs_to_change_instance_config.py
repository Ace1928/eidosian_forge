from __future__ import absolute_import, division, print_function
import copy
import datetime
import os
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _needs_to_change_instance_config(self, key):
    if key not in self.config:
        return False
    if key == 'config':
        old_configs = dict(self.old_sections.get(key, None) or {})
        for k, v in self.config['config'].items():
            if k not in old_configs:
                return True
            if old_configs[k] != v:
                return True
        return False
    else:
        old_configs = self.old_sections.get(key, {})
        return self.config[key] != old_configs