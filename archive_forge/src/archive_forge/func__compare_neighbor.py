from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.bgp_global import (
def _compare_neighbor(self, want, have):
    parsers = ['neighbor.advertisement_interval', 'neighbor.allowas_in', 'neighbor.as_override', 'neighbor.attribute_unchanged.as_path', 'neighbor.attribute_unchanged.med', 'neighbor.attribute_unchanged.next_hop', 'neighbor.capability_dynamic', 'neighbor.capability_orf', 'neighbor.default_originate', 'neighbor.description', 'neighbor.disable_capability_negotiation', 'neighbor.disable_connected_check', 'neighbor.disable_send_community', 'neighbor.distribute_list', 'neighbor.ebgp_multihop', 'neighbor.filter_list', 'neighbor.local_as', 'neighbor.maximum_prefix', 'neighbor.nexthop_self', 'neighbor.override_capability', 'neighbor.passive', 'neighbor.password', 'neighbor.peer_group_name', 'neighbor.port', 'neighbor.prefix_list', 'neighbor.remote_as', 'neighbor.remove_private_as', 'neighbor.route_map', 'neighbor.route_reflector_client', 'neighbor.route_server_client', 'neighbor.shutdown', 'neighbor.soft_reconfiguration', 'neighbor.strict_capability_match', 'neighbor.unsuppress_map', 'neighbor.update_source', 'neighbor.weight', 'neighbor.ttl_security', 'neighbor.timers', 'network.backdoor', 'network.route_map']
    wneigh = want.pop('neighbor', {})
    hneigh = have.pop('neighbor', {})
    self._compare_neigh_lists(wneigh, hneigh)
    for name, entry in iteritems(wneigh):
        for k, v in entry.items():
            if k == 'address':
                continue
            if hneigh.get(name):
                h = {'address': name, k: hneigh[name].pop(k, {})}
            else:
                h = {}
            self.compare(parsers=parsers, want={'as_number': want['as_number'], 'neighbor': {'address': name, k: v}}, have={'as_number': want['as_number'], 'neighbor': h})
    for name, entry in iteritems(hneigh):
        if name not in wneigh.keys():
            if self._check_af(name):
                msg = 'Use the _bgp_address_family module to delete the address_family under neighbor {0}, before replacing/deleting the neighbor.'.format(name)
                self._module.fail_json(msg=msg)
            else:
                self.commands.append('delete protocols bgp ' + str(have['as_number']) + ' neighbor ' + name)
                continue
        for k, v in entry.items():
            self.compare(parsers=parsers, want={}, have={'as_number': have['as_number'], 'neighbor': {'address': name, k: v}})