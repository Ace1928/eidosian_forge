from __future__ import absolute_import, division, print_function
import collections
import os
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import Version
def addremovekey(self, command):
    """Add or remove key based on command"""
    admin = self._module.params.get('admin')
    conn = get_connection(self._module)
    if admin:
        conn.send_command('admin')
    out = conn.send_command(command, prompt='yes/no', answer='yes')
    if admin:
        conn.send_command('exit')
    return out