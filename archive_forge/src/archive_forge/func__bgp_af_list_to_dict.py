from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.bgp_address_family import (
def _bgp_af_list_to_dict(self, entry):
    for name, proc in iteritems(entry):
        if 'address_family' in proc:
            af_dict = {}
            for entry in proc.get('address_family'):
                if 'networks' in entry:
                    network_dict = {}
                    for n_entry in entry.get('networks', []):
                        network_dict.update({n_entry['prefix']: n_entry})
                    entry['networks'] = network_dict
                if 'aggregate_address' in entry:
                    agg_dict = {}
                    for a_entry in entry.get('aggregate_address', []):
                        agg_dict.update({a_entry['prefix']: a_entry})
                    entry['aggregate_address'] = agg_dict
                if 'redistribute' in entry:
                    redis_dict = {}
                    for r_entry in entry.get('redistribute', []):
                        proto_key = r_entry.get('protocol', 'table')
                        redis_dict.update({proto_key: r_entry})
                    entry['redistribute'] = redis_dict
            for af in proc.get('address_family'):
                af_dict.update({af['afi']: af})
            proc['address_family'] = af_dict
        if 'neighbors' in proc:
            neigh_dict = {}
            for entry in proc.get('neighbors', []):
                neigh_dict.update({entry['neighbor_address']: entry})
            proc['neighbors'] = neigh_dict
            self._bgp_af_list_to_dict(proc['neighbors'])