from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_alert_policy_is_present(self):
    """
        Ensures that the alert policy is present
        :return: (changed, policy)
                 changed: A flag representing if anything is modified
                 policy: the created/updated alert policy
        """
    changed = False
    p = self.module.params
    policy_name = p.get('name')
    if not policy_name:
        self.module.fail_json(msg='Policy name is a required')
    policy = self._alert_policy_exists(policy_name)
    if not policy:
        changed = True
        policy = None
        if not self.module.check_mode:
            policy = self._create_alert_policy()
    else:
        changed_u, policy = self._ensure_alert_policy_is_updated(policy)
        if changed_u:
            changed = True
    return (changed, policy)