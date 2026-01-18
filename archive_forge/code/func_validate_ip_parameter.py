from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
def validate_ip_parameter(self, sds_ip_list):
    """Validate the input parameters"""
    if sds_ip_list is None or len(sds_ip_list) == 0:
        error_msg = "Provide valid values for sds_ip_list as 'ip' and 'role' for Create/Modify operations."
        LOG.error(error_msg)
        self.module.fail_json(msg=error_msg)