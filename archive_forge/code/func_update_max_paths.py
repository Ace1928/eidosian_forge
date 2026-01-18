from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp_af.bgp_af import Bgp_afArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
def update_max_paths(self, data):
    for conf in data:
        afs = conf.get('address_family', [])
        if afs:
            for af in afs:
                max_path = {}
                ebgp = af.get('ebgp', None)
                if ebgp:
                    af.pop('ebgp')
                    max_path['ebgp'] = ebgp
                ibgp = af.get('ibgp', None)
                if ibgp:
                    af.pop('ibgp')
                    max_path['ibgp'] = ibgp
                if max_path:
                    af['max_path'] = max_path