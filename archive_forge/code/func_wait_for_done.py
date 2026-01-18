from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.community.general.plugins.module_utils.opennebula import flatten, render
def wait_for_done(module, client, vm, wait_timeout):
    return wait_for_state(module, client, vm, wait_timeout, lambda state, lcm_state: state in [VM_STATES.index('DONE')])