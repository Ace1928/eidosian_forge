from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def __derive_static_route_next_hop_config_key_match_op(key_set, command, exist_conf):
    bh = command['index'].get('blackhole', None)
    itf = command['index'].get('interface', None)
    nv = command['index'].get('nexthop_vrf', None)
    nh = command['index'].get('next_hop', None)
    conf_bh = exist_conf['index'].get('blackhole', None)
    conf_itf = exist_conf['index'].get('interface', None)
    conf_nv = exist_conf['index'].get('nexthop_vrf', None)
    conf_nh = exist_conf['index'].get('next_hop', None)
    if bh == conf_bh and itf == conf_itf and (nv == conf_nv) and (nh == conf_nh):
        return True
    else:
        return False