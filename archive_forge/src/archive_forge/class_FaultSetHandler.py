from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell import (
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
from ansible.module_utils.basic import AnsibleModule
class FaultSetHandler:

    def handle(self, fault_set_obj, fault_set_params):
        fault_set_obj.validate_parameters(fault_set_params=fault_set_params)
        pd_id = None
        if fault_set_params['protection_domain_id'] or fault_set_params['protection_domain_name']:
            pd_id = fault_set_obj.get_protection_domain(protection_domain_id=fault_set_params['protection_domain_id'], protection_domain_name=fault_set_params['protection_domain_name'])['id']
        fault_set_details = fault_set_obj.get_fault_set(fault_set_id=fault_set_params['fault_set_id'], fault_set_name=fault_set_params['fault_set_name'], protection_domain_id=pd_id)
        FaultSetCreateHandler().handle(fault_set_obj, fault_set_params, fault_set_details, pd_id)