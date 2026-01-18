from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.bgp_af.bgp_af import Bgp_afArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
def update_redis_data(self, objs, af_redis_data):
    if not (af_redis_data or objs):
        return
    for conf in objs:
        vrf_name = conf['vrf_name']
        raw_af_redis_data = next((e_af_redis for e_af_redis in af_redis_data if vrf_name in e_af_redis), None)
        if not raw_af_redis_data:
            continue
        norm_af_redis_data = self.normalize_af_redis_params(raw_af_redis_data[vrf_name])
        if norm_af_redis_data:
            if 'address_family' in conf:
                afs = conf['address_family']
                if not afs:
                    continue
                for e_af in afs:
                    if 'afi' in e_af:
                        afi = e_af['afi']
                        redis_arr = []
                        for e_redis_data in norm_af_redis_data:
                            if self.check_afi(afi, e_redis_data):
                                e_redis_data.pop('afi')
                                redis_arr.append(e_redis_data)
                        e_af.update({'redistribute': redis_arr})
            else:
                addr_fams = []
                for e_norm_af_redis in norm_af_redis_data:
                    afi = e_norm_af_redis['afi']
                    e_norm_af_redis.pop('afi')
                    mat_addr_fam = next((each_addr_fam for each_addr_fam in addr_fams if each_addr_fam['afi'] == afi), None)
                    if mat_addr_fam:
                        mat_addr_fam['redistribute'].append(e_norm_af_redis)
                    else:
                        addr_fams.append({'redistribute': [e_norm_af_redis], 'afi': afi})
                if addr_fams:
                    conf.update({'address_family': addr_fams})