from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def validate_adv_host_mgmt_enabled_check(self, host_dict):
    """
        Validate adv_host_mgmt_enabled check
        :param host_dict: Host access data
        :return None
        """
    host_dict_keys_set = set(host_dict.keys())
    adv_host_mgmt_enabled_true_set = {'host_name', 'host_id', 'ip_address'}
    adv_host_mgmt_enabled_false_set = {'host_name', 'subnet', 'domain', 'netgroup', 'ip_address'}
    adv_host_mgmt_enabled_true_diff = host_dict_keys_set - adv_host_mgmt_enabled_true_set
    adv_host_mgmt_enabled_false_diff = host_dict_keys_set - adv_host_mgmt_enabled_false_set
    if self.module.params['adv_host_mgmt_enabled'] and adv_host_mgmt_enabled_true_diff != set():
        msg = "If 'adv_host_mgmt_enabled' is true then host access should only have  %s" % adv_host_mgmt_enabled_true_set
        LOG.error(msg)
        self.module.fail_json(msg=msg)
    elif not self.module.params['adv_host_mgmt_enabled'] and adv_host_mgmt_enabled_false_diff != set():
        msg = "If 'adv_host_mgmt_enabled' is false then host access should only have  %s" % adv_host_mgmt_enabled_false_set
        LOG.error(msg)
        self.module.fail_json(msg=msg)