from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def action_compare(module, existing_snapshots):
    command = 'show snapshot compare {0} {1}'.format(module.params['snapshot1'], module.params['snapshot2'])
    if module.params['compare_option']:
        command += ' {0}'.format(module.params['compare_option'])
    body = execute_show_command(command, module)[0]
    return body