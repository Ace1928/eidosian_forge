from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell\
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.powerflex_base \
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell.libraries.configuration \
import copy
class SDSExitHandler:

    def handle(self, sds_obj, sds_details):
        if sds_details:
            sds_obj.result['sds_details'] = sds_obj.show_output(sds_id=sds_details['id'])
        else:
            sds_obj.result['sds_details'] = None
        sds_obj.module.exit_json(**sds_obj.result)