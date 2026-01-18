from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp_af.bgp_af import Bgp_afArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
def update_route_advertise_list(self, data):
    for conf in data:
        afs = conf.get('address_family', [])
        if afs:
            for af in afs:
                rt_adv_lst = []
                route_advertise_list = af.get('route_advertise_list', None)
                if route_advertise_list:
                    for rt in route_advertise_list:
                        rt_adv_dict = {}
                        advertise_afi = rt['advertise-afi-safi'].split(':')[1].split('_')[0].lower()
                        route_map_config = rt['config']
                        route_map = route_map_config.get('route-map', None)
                        if advertise_afi:
                            rt_adv_dict['advertise_afi'] = advertise_afi
                        if route_map:
                            rt_adv_dict['route_map'] = route_map[0]
                        if rt_adv_dict and rt_adv_dict not in rt_adv_lst:
                            rt_adv_lst.append(rt_adv_dict)
                    af['route_advertise_list'] = rt_adv_lst