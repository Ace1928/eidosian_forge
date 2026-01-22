from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
class SDSDeleteHandler:

    def handle(self, sds_obj, sds_params, sds_details):
        if sds_params['state'] == 'absent' and sds_details:
            sds_details = sds_obj.delete_sds(sds_details['id'])
            sds_obj.result['changed'] = True
        SDSExitHandler().handle(sds_obj, sds_details)