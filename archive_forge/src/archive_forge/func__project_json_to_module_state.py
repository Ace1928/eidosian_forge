from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.lxd import (
from ansible.module_utils.basic import AnsibleModule
import os
@staticmethod
def _project_json_to_module_state(resp_json):
    if resp_json['type'] == 'error':
        return 'absent'
    return 'present'