from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import run_commands
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import command_list_str_to_dict
def transform_commands_dict(module, commands_dict):
    transform = EntityCollection(module, dict(command=dict(key=True), output=dict(), prompt=dict(type='list'), answer=dict(type='list'), newline=dict(type='bool', default=True), sendonly=dict(type='bool', default=False), check_all=dict(type='bool', default=False)))
    return transform(commands_dict)