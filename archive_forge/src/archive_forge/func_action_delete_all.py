from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def action_delete_all(module, existing_snapshots):
    commands = list()
    if existing_snapshots:
        commands.append('snapshot delete all')
    return commands