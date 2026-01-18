from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def validate_create(self, protection_domain_id, sds_ip_list, sds_ip_state, sds_name, sds_id, sds_new_name, rmcache_enabled=None, rmcache_size=None, fault_set_id=None):
    if sds_name is None or len(sds_name.strip()) == 0:
        error_msg = 'Please provide valid sds_name value for creation of SDS.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
    if protection_domain_id is None:
        error_msg = 'Protection Domain is a mandatory parameter for creating an SDS. Please enter a valid value.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
    if sds_ip_list is None or len(sds_ip_list) == 0:
        error_msg = 'Please provide valid sds_ip_list values for creation of SDS.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
    if sds_ip_state is not None and sds_ip_state != 'present-in-sds':
        error_msg = 'Incorrect IP state given for creation of SDS.'
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)
    if sds_id:
        error_msg = 'Creation of SDS is allowed using sds_name only, sds_id given.'
        LOG.info(error_msg)
        self.module.fail_json(msg=error_msg)