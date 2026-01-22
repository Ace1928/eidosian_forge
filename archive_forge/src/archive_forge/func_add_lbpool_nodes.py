from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from time import sleep
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def add_lbpool_nodes(self, alias, location, lb_id, pool_id, nodes_to_add):
    """
        Add nodes to the provided pool
        :param alias: the account alias
        :param location: the datacenter the load balancer resides in
        :param lb_id: the id string of the load balancer
        :param pool_id: the id string of the pool
        :param nodes_to_add: a list of dictionaries containing the nodes to add
        :return: (changed, result) -
            changed: Boolean whether a change was made
            result: The result from the CLC API call
        """
    changed = False
    result = {}
    nodes = self._get_lbpool_nodes(alias, location, lb_id, pool_id)
    for node in nodes_to_add:
        if not node.get('status'):
            node['status'] = 'enabled'
        if node not in nodes:
            changed = True
            nodes.append(node)
    if changed is True and (not self.module.check_mode):
        result = self.set_loadbalancernodes(alias, location, lb_id, pool_id, nodes)
    return (changed, result)