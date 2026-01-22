from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
def _get_fw_type(self, afi):
    """
        This function returns the firewall rule-set type based on IP address.
        :param afi: address type
        :return: rule-set type.
        """
    return 'ipv6-name' if afi == 'ipv6' else 'name'