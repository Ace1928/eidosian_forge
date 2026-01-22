from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.six.moves.urllib.parse import urlparse
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_firewall_policy_is_absent(self, source_account_alias, location, firewall_dict):
    """
        Ensures that a given firewall policy is removed if present
        :param source_account_alias: the source account alias for the firewall policy
        :param location: datacenter of the firewall policy
        :param firewall_dict: firewall policy to delete
        :return: (changed, firewall_policy_id, response)
            changed: flag for if a change occurred
            firewall_policy_id: the firewall policy id that was deleted
            response: response from CLC API call
        """
    changed = False
    response = []
    firewall_policy_id = firewall_dict.get('firewall_policy_id')
    result = self._get_firewall_policy(source_account_alias, location, firewall_policy_id)
    if result:
        if not self.module.check_mode:
            response = self._delete_firewall_policy(source_account_alias, location, firewall_policy_id)
        changed = True
    return (changed, firewall_policy_id, response)