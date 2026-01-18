from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.stp.stp import StpArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def update_stp(self, data):
    config_dict = {}
    if data:
        config_dict['global'] = self.update_global(data)
        config_dict['interfaces'] = self.update_interfaces(data)
        config_dict['mstp'] = self.update_mstp(data)
        config_dict['pvst'] = self.update_pvst(data)
        config_dict['rapid_pvst'] = self.update_rapid_pvst(data)
    return config_dict