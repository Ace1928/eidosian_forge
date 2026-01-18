from __future__ import (absolute_import, division, print_function)
import os
import tempfile
import copy
from ansible_collections.dellemc.openmanage.plugins.module_utils.dellemc_idrac import iDRACConnection, idrac_auth_params
from ansible.module_utils.basic import AnsibleModule
def view_storage(idrac, module):
    idrac.get_entityjson()
    storage_status = idrac.config_mgr.RaidHelper.view_storage(controller=module.params['controller_id'], virtual_disk=module.params['volume_id'])
    if storage_status['Status'] == 'Failed':
        module.fail_json(msg='Failed to fetch storage details', storage_status=storage_status)
    return storage_status