from __future__ import absolute_import, division, print_function
import copy
import datetime
import os
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _needs_to_apply_instance_configs(self):
    for param in set(CONFIG_PARAMS) - set(CONFIG_CREATION_PARAMS):
        if self._needs_to_change_instance_config(param):
            return True
    return False