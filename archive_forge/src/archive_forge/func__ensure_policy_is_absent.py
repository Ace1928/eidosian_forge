from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_policy_is_absent(self, p):
    """
        Makes sure that a policy is absent
        :param p: dictionary of policy name
        :return: tuple of if a deletion occurred and the name of the policy that was deleted
        """
    changed = False
    if self._policy_exists(policy_name=p['name']):
        changed = True
        if not self.module.check_mode:
            self._delete_policy(p)
    return (changed, None)