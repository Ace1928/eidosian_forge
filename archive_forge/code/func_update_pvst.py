from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.stp.stp import StpArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def update_pvst(self, data):
    pvst_list = []
    pvst = data.get('openconfig-spanning-tree-ext:pvst', None)
    if pvst:
        vlans = pvst.get('vlans', None)
        if vlans:
            vlans_list = self.get_vlans_list(vlans)
            if vlans_list:
                pvst_list = vlans_list
    return pvst_list