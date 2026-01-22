from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_memory(clc, module):
    """
        Find or validate the Memory value by calling the CLC API
        :param clc: clc-sdk instance to use
        :param module: module to validate
        :return: Int value for Memory
        """
    memory = module.params.get('memory')
    group_id = module.params.get('group_id')
    alias = module.params.get('alias')
    state = module.params.get('state')
    if not memory and state == 'present':
        group = clc.v2.Group(id=group_id, alias=alias)
        if group.Defaults('memory'):
            memory = group.Defaults('memory')
        else:
            module.fail_json(msg=str("Can't determine a default memory value. Please provide a value for memory."))
    return memory