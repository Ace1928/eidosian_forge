from __future__ import absolute_import, division, print_function
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell import (
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
from ansible.module_utils.basic import AnsibleModule
class FaultSetExitHandler:

    def handle(self, fault_set_obj, fault_set_details):
        fault_set_obj.result['fault_set_details'] = fault_set_details
        if fault_set_details:
            fault_set_obj.result['fault_set_details']['protectionDomainName'] = fault_set_obj.get_protection_domain(protection_domain_id=fault_set_details['protectionDomainId'])['name']
            fault_set_obj.result['fault_set_details']['SDS'] = fault_set_obj.get_associated_sds(fault_set_id=fault_set_details['id'])
        fault_set_obj.module.exit_json(**fault_set_obj.result)