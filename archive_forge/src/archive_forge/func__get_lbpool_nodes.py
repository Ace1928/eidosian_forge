from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _get_lbpool_nodes(self, alias, location, lb_id, pool_id):
    """
        Return the list of nodes available to the provided load balancer pool
        :param alias: the account alias
        :param location: the datacenter the load balancer resides in
        :param lb_id: the id string of the load balancer
        :param pool_id: the id string of the pool
        :return: result: The list of nodes
        """
    result = None
    try:
        result = self.clc.v2.API.Call('GET', '/v2/sharedLoadBalancers/%s/%s/%s/pools/%s/nodes' % (alias, location, lb_id, pool_id))
    except APIFailedResponse as e:
        self.module.fail_json(msg='Unable to fetch list of available nodes for load balancer pool id: {0}. {1}'.format(pool_id, str(e.response_text)))
    return result