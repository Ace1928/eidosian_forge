from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.bgp_address_family import (
def _compare_neighbor_lists(self, want, have):
    """Compare neighbor list of dict"""
    neig_parses = ['peer_group', 'peer_group_name', 'local_as', 'remote_as', 'activate', 'additional_paths', 'advertises.additional_paths', 'advertises.best_external', 'advertises.diverse_path', 'advertise_map', 'advertisement_interval', 'aigp', 'aigp.send.cost_community', 'aigp.send.med', 'allow_policy', 'allowas_in', 'as_override', 'bmp_activate', 'capability', 'cluster_id', 'default_originate', 'default_originate.route_map', 'description', 'disable_connected_check', 'ebgp_multihop', 'distribute_list', 'dmzlink_bw', 'filter_list', 'fall_over.bfd', 'fall_over.route_map', 'ha_mode', 'inherit', 'internal_vpn_client', 'log_neighbor_changes', 'maximum_prefix', 'nexthop_self.set', 'nexthop_self.all', 'next_hop_unchanged.set', 'next_hop_unchanged.allpaths', 'password_options', 'path_attribute.discard', 'path_attribute.treat_as_withdraw', 'route_maps', 'remove_private_as.set', 'remove_private_as.all', 'remove_private_as.replace_as', 'route_reflector_client', 'route_server_client', 'send_community.set', 'send_community.both', 'send_community.extended', 'send_community.standard', 'shutdown', 'slow_peer_options.detection', 'slow_peer_options.split_update_group', 'soft_reconfiguration', 'soo', 'timers', 'transport.connection_mode', 'transport.multi_session', 'transport.path_mtu_discovery', 'ttl_security', 'unsuppress_map', 'version', 'weight']
    for name, w_neighbor in want.items():
        have_nbr = have.pop(name, {})
        self.compare(parsers=neig_parses, want=w_neighbor, have=have_nbr)
        for i in ['route_maps', 'prefix_lists']:
            want_route_or_prefix = w_neighbor.pop(i, {})
            have_route_or_prefix = have_nbr.pop(i, {})
            if want_route_or_prefix:
                for k_rmps, w_rmps in want_route_or_prefix.items():
                    have_rmps = have_route_or_prefix.pop(k_rmps, {})
                    w_rmps['neighbor_address'] = w_neighbor.get('neighbor_address')
                    if have_rmps:
                        have_rmps['neighbor_address'] = have_nbr.get('neighbor_address')
                        have_rmps = {i: have_rmps}
                    self.compare(parsers=[i], want={i: w_rmps}, have=have_rmps)
    for name, h_neighbor in have.items():
        self.compare(parsers='neighbor_address', want={}, have=h_neighbor)