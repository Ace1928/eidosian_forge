from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def validate_pause(self, params):
    if params['pause'] is not None and (not params['pair_id'] and (not params['pair_name'])):
        self.module.fail_json(msg='Specify either pair_id or pair_name to perform pause or resume of initial copy')