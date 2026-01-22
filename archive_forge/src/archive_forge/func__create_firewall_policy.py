from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.six.moves.urllib.parse import urlparse
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _create_firewall_policy(self, source_account_alias, location, firewall_dict):
    """
        Creates the firewall policy for the given account alias
        :param source_account_alias: the source account alias for the firewall policy
        :param location: datacenter of the firewall policy
        :param firewall_dict: dictionary of request parameters for firewall policy
        :return: response from CLC API call
        """
    payload = {'destinationAccount': firewall_dict.get('destination_account_alias'), 'source': firewall_dict.get('source'), 'destination': firewall_dict.get('destination'), 'ports': firewall_dict.get('ports')}
    try:
        response = self.clc.v2.API.Call('POST', '/v2-experimental/firewallPolicies/%s/%s' % (source_account_alias, location), payload)
    except APIFailedResponse as e:
        return self.module.fail_json(msg='Unable to create firewall policy. %s' % str(e.response_text))
    return response