from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.six.moves.urllib.parse import urlparse
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _compare_get_request_with_dict(response, firewall_dict):
    """
        Helper method to compare the json response for getting the firewall policy with the request parameters
        :param response: response from the get method
        :param firewall_dict: dictionary of request parameters for firewall policy
        :return: changed: Boolean that returns true if there are differences between
                          the response parameters and the playbook parameters
        """
    changed = False
    response_dest_account_alias = response.get('destinationAccount')
    response_enabled = response.get('enabled')
    response_source = response.get('source')
    response_dest = response.get('destination')
    response_ports = response.get('ports')
    request_dest_account_alias = firewall_dict.get('destination_account_alias')
    request_enabled = firewall_dict.get('enabled')
    if request_enabled is None:
        request_enabled = True
    request_source = firewall_dict.get('source')
    request_dest = firewall_dict.get('destination')
    request_ports = firewall_dict.get('ports')
    if response_dest_account_alias and str(response_dest_account_alias) != str(request_dest_account_alias) or response_enabled != request_enabled or (response_source and response_source != request_source) or (response_dest and response_dest != request_dest) or (response_ports and response_ports != request_ports):
        changed = True
    return changed