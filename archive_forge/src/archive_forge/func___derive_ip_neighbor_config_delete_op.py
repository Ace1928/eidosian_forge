from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
from ansible.module_utils.connection import ConnectionError
def __derive_ip_neighbor_config_delete_op(key_set, command, exist_conf):
    new_conf = exist_conf
    if 'ipv4_arp_timeout' in command:
        new_conf['ipv4_arp_timeout'] = IP_NEIGH_CONFIG_DEFAULT['ipv4_arp_timeout']
    if 'ipv4_drop_neighbor_aging_time' in command:
        new_conf['ipv4_drop_neighbor_aging_time'] = IP_NEIGH_CONFIG_DEFAULT['ipv4_drop_neighbor_aging_time']
    if 'ipv6_drop_neighbor_aging_time' in command:
        new_conf['ipv6_drop_neighbor_aging_time'] = IP_NEIGH_CONFIG_DEFAULT['ipv6_drop_neighbor_aging_time']
    if 'ipv6_nd_cache_expiry' in command:
        new_conf['ipv6_nd_cache_expiry'] = IP_NEIGH_CONFIG_DEFAULT['ipv6_nd_cache_expiry']
    if 'num_local_neigh' in command:
        new_conf['num_local_neigh'] = IP_NEIGH_CONFIG_DEFAULT['num_local_neigh']
    return (True, new_conf)