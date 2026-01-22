from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.lxd import (
from ansible.module_utils.basic import AnsibleModule
import os
def _needs_to_apply_project_configs(self):
    return self._needs_to_change_project_config('config') or self._needs_to_change_project_config('description')