from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.facts.facts import Facts
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.rm_templates.bgp_templates import (
def _compare_ngs(self, want, have):
    """Leverages the base class `compare()` method and
        populates the list of commands to be run by comparing
        the `want` and `have` data with the `parsers` defined
        for the Bgp_global neighbor resource.
        """
    neighbor_parsers = ['use.neighbor_group', 'use.session_group', 'advertisement_interval', 'bfd_fast_detect_disable', 'bfd_fast_detect_strict_mode', 'bfd_fast_detect_set', 'bfd_nbr_minimum_interval', 'bfd_nbr_multiplier', 'bmp_activate', 'dmz_link_bandwidth', 'dmz_link_bandwidth_inheritance_disable', 'neighbor_description', 'neighbor_cluster_id', 'dscp', 'ebgp_multihop_value', 'ebgp_multihop_mpls', 'ebgp_recv_extcommunity_dmz', 'ebgp_recv_extcommunity_dmz_set', 'ebgp_send_extcommunity_dmz', 'ebgp_send_extcommunity_dmz_set', 'ebgp_send_extcommunity_dmz_cumulatie', 'egress_engineering', 'egress_engineering_set', 'idle_watch_time', 'internal_vpn_client', 'ignore_connected_check', 'ignore_connected_check_set', 'neighbor_enforce_first_as_disable', 'neighbor_graceful_restart_restart_time', 'neighbor_graceful_restart_stalepath_time', 'keychain', 'keychain_name', 'local_as_inheritance_disable', 'local_as', 'local', 'local_address', 'origin_as', 'password_inheritance_disable', 'password_encrypted', 'peer_set', 'precedence', 'remote_as', 'remote_as_list', 'receive_buffer_size', 'send_buffer_size', 'session_open_mode', 'neighbor_shutdown', 'neighbor_shutdown_inheritance_disable', 'neighbor_tcp_mss', 'neighbor_tcp_mss_inheritance_disable', 'neighbor_timers_keepalive', 'update_source', 'neighbor_ttl_security_inheritance_disable', 'neighbor_ttl_security', 'neighbor_graceful_maintenance_set', 'neighbor_graceful_maintenance_activate', 'neighbor_graceful_maintenance_activate_inheritance_disable', 'neighbor_graceful_maintenance_as_prepends', 'neighbor_graceful_maintenance_local_preference_disable', 'neighbor_graceful_maintenance_local_preference', 'neighbor_graceful_maintenance_as_prepends_value', 'neighbor_capability_additional_paths_send', 'neighbor_capability_additional_paths_send_disable', 'neighbor_capability_additional_paths_rcv_disable', 'neighbor_capability_additional_paths_rcv', 'neighbor_capability_suppress_four_byte_AS', 'neighbor_capability_suppress_all', 'neighbor_capability_suppress_all_inheritance_disable', 'neighbor_log_message_in_value', 'neighbor_log_message_in_disable', 'neighbor_log_message_in_inheritance_disable', 'neighbor_log_message_out_value', 'neighbor_log_message_out_disable', 'neighbor_log_message_out_inheritance_disable', 'neighbor_update_in_filtering_attribute_filter_group', 'neighbor_update_in_filtering_logging_disable', 'neighbor_update_in_filtering_message_buffers']
    want_nbr = want.get('neighbor', {})
    have_nbr = have.get('neighbor', {})
    for name, entry in iteritems(want_nbr):
        have = have_nbr.pop(name, {})
        begin = len(self.commands)
        self.compare(parsers=neighbor_parsers, want=entry, have=have)
        if self.state in ['replaced', 'overridden']:
            self.sort_commands(begin)
        self._compare_af(want=entry, have=have)
        name = entry.get('name', '')
        if len(self.commands) != begin:
            self.commands.insert(begin, self._tmplt.render({'name': name}, 'neighbor_group', False))