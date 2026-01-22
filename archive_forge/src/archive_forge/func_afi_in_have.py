from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def afi_in_have(self, have, w_item):
    """
        This functions checks for the afi
        list in have
        :param have:
        :param w_item:
        :return:
        """
    if have:
        for h in have:
            af = h.get('address_families') or []
        for item in af:
            if w_item['afi'] == item['afi']:
                return True
    return False