from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _alert_policy_exists(self, policy_name):
    """
        Check to see if an alert policy exists
        :param policy_name: name of the alert policy
        :return: boolean of if the policy exists
        """
    result = False
    for policy_id in self.policy_dict:
        if self.policy_dict.get(policy_id).get('name') == policy_name:
            result = self.policy_dict.get(policy_id)
    return result