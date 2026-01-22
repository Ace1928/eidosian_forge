from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _get_loadbalancer_list(self, alias, location):
    """
        Retrieve a list of loadbalancers
        :param alias: Alias for account
        :param location: Datacenter
        :return: JSON data for all loadbalancers at datacenter
        """
    result = None
    try:
        result = self.clc.v2.API.Call('GET', '/v2/sharedLoadBalancers/%s/%s' % (alias, location))
    except APIFailedResponse as e:
        self.module.fail_json(msg='Unable to fetch load balancers for account: {0}. {1}'.format(alias, str(e.response_text)))
    return result