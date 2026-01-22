from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
@staticmethod
def any_rmap_inst_in_have(conf_map_name, have):
    """In the current configuration on the target device, determine if there
        is at least one configuration "statement" for the specified route map name
        from the input playbook request."""
    for cfg_route_map in have:
        if cfg_route_map.get('map_name'):
            if cfg_route_map['map_name'] == conf_map_name:
                return True
    return False