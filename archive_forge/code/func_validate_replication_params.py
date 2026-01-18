from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_replication_params(self, replication_params):
    """ Validate replication params """
    if not replication_params:
        errormsg = 'Please specify replication_params to enable replication.'
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    if replication_params['destination_pool_id'] is not None and replication_params['destination_pool_name'] is not None:
        errormsg = "'destination_pool_id' and 'destination_pool_name' is mutually exclusive."
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)
    self.validate_rpo(replication_params)
    if replication_params['replication_type'] == 'remote' and replication_params['remote_system'] is None:
        errormsg = "Remote_system is required together with 'remote' replication_type"
        LOG.error(errormsg)
        self.module.fail_json(msg=errormsg)