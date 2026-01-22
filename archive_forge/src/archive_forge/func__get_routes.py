from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _get_routes(self, lst):
    """
        This function returns the list of routes
        :param lst: list of address families
        :return: list of routes
        """
    r_list = []
    for item in lst:
        af = item['address_families']
        for element in af:
            routes = element.get('routes') or []
            for r in routes:
                r_list.append(r)
    return r_list