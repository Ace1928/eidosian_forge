from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _get_r_sets(self, item, type='rule_sets'):
    """
        This function returns the list of rule-sets/rules.
        :param item: config dictionary.
        :param type: rule_sets/rule/r_list.
        :return: list of rule-sets/rules.
        """
    rs_list = []
    r_sets = item[type]
    if r_sets:
        for rs in r_sets:
            rs_list.append(rs)
    return rs_list