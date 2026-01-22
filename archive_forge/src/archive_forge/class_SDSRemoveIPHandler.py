from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
class SDSRemoveIPHandler:

    def handle(self, sds_obj, sds_params, sds_details, sds_ip_list):
        if sds_params['state'] == 'present' and sds_details:
            remove_ip_changed = False
            if sds_params['sds_ip_state'] == 'absent-in-sds':
                sds_obj.validate_ip_parameter(sds_ip_list)
                ips_to_remove = sds_obj.identify_ip_role_remove(sds_ip_list, sds_details, sds_params['sds_ip_state'])
                if ips_to_remove:
                    remove_ip_changed = sds_obj.remove_ip(sds_details['id'], ips_to_remove)
                if remove_ip_changed:
                    sds_obj.result['changed'] = True
        SDSDeleteHandler().handle(sds_obj, sds_params, sds_details)