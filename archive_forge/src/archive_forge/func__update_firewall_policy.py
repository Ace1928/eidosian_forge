from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.six.moves.urllib.parse import urlparse
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _update_firewall_policy(self, source_account_alias, location, firewall_policy_id, firewall_dict):
    """
        Updates a firewall policy for a given datacenter and account alias
        :param source_account_alias: the source account alias for the firewall policy
        :param location: datacenter of the firewall policy
        :param firewall_policy_id: firewall policy id to update
        :param firewall_dict: dictionary of request parameters for firewall policy
        :return: response: response from CLC API call
        """
    try:
        response = self.clc.v2.API.Call('PUT', '/v2-experimental/firewallPolicies/%s/%s/%s' % (source_account_alias, location, firewall_policy_id), firewall_dict)
    except APIFailedResponse as e:
        return self.module.fail_json(msg='Unable to update the firewall policy id : {0}. {1}'.format(firewall_policy_id, str(e.response_text)))
    return response