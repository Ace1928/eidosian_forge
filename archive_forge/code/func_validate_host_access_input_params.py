from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_host_access_input_params(self):
    """
        Validate host access params
        :return None
        """
    for param in list(self.host_param_mapping.keys()):
        if self.module.params[param] and (not self.module.params['host_state'] or self.module.params['adv_host_mgmt_enabled'] is None):
            msg = "'host_state' and 'adv_host_mgmt_enabled' is required along with: %s" % param
            LOG.error(msg)
            self.module.fail_json(msg=msg)
        elif self.module.params[param]:
            for host_dict in self.module.params[param]:
                host_dict = {k: v for k, v in host_dict.items() if v}
                self.validate_adv_host_mgmt_enabled_check(host_dict)
                self.validate_host_access_data(host_dict)