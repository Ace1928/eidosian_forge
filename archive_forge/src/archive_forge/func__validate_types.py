from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _validate_types(module):
    """
        Validate that type and storage_type are set appropriately, and fail if not
        :param module: the module to validate
        :return: none
        """
    state = module.params.get('state')
    server_type = module.params.get('type').lower() if module.params.get('type') else None
    storage_type = module.params.get('storage_type').lower() if module.params.get('storage_type') else None
    if state == 'present':
        if server_type == 'standard' and storage_type not in ('standard', 'premium'):
            module.fail_json(msg=str("Standard VMs must have storage_type = 'standard' or 'premium'"))
        if server_type == 'hyperscale' and storage_type != 'hyperscale':
            module.fail_json(msg=str("Hyperscale VMs must have storage_type = 'hyperscale'"))