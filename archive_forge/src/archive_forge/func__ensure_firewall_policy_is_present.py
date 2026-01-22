from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.six.moves.urllib.parse import urlparse
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _ensure_firewall_policy_is_present(self, source_account_alias, location, firewall_dict):
    """
        Ensures that a given firewall policy is present
        :param source_account_alias: the source account alias for the firewall policy
        :param location: datacenter of the firewall policy
        :param firewall_dict: dictionary of request parameters for firewall policy
        :return: (changed, firewall_policy_id, firewall_policy)
            changed: flag for if a change occurred
            firewall_policy_id: the firewall policy id that was created/updated
            firewall_policy: The firewall_policy object
        """
    firewall_policy = None
    firewall_policy_id = firewall_dict.get('firewall_policy_id')
    if firewall_policy_id is None:
        if not self.module.check_mode:
            response = self._create_firewall_policy(source_account_alias, location, firewall_dict)
            firewall_policy_id = self._get_policy_id_from_response(response)
        changed = True
    else:
        firewall_policy = self._get_firewall_policy(source_account_alias, location, firewall_policy_id)
        if not firewall_policy:
            return self.module.fail_json(msg='Unable to find the firewall policy id : {0}'.format(firewall_policy_id))
        changed = self._compare_get_request_with_dict(firewall_policy, firewall_dict)
        if not self.module.check_mode and changed:
            self._update_firewall_policy(source_account_alias, location, firewall_policy_id, firewall_dict)
    if changed and firewall_policy_id:
        firewall_policy = self._wait_for_requests_to_complete(source_account_alias, location, firewall_policy_id)
    return (changed, firewall_policy_id, firewall_policy)