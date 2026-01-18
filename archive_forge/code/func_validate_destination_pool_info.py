from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_destination_pool_info(self, replication):
    if replication['destination_pool_id'] is not None and replication['destination_pool_name'] is not None:
        errormsg = "'destination_pool_id' and 'destination_pool_name' is mutually exclusive."
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    if replication['destination_pool_id'] is None and replication['destination_pool_name'] is None:
        errormsg = "Either 'destination_pool_id' or 'destination_pool_name' is required."
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)