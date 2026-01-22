from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def CreateNetworkResourceFromArgs(messages, network_ref, network_args, support_firewall_order):
    """Creates a new network resource from flag arguments."""
    network = messages.Network(name=network_ref.Name(), description=network_args.description)
    if network_args.subnet_mode == 'legacy':
        network.IPv4Range = network_args.range
    elif network_args.subnet_mode == 'custom':
        network.autoCreateSubnetworks = False
    else:
        network.autoCreateSubnetworks = True
    network.routingConfig = messages.NetworkRoutingConfig()
    network.routingConfig.routingMode = messages.NetworkRoutingConfig.RoutingModeValueValuesEnum(network_args.bgp_routing_mode.upper())
    if getattr(network_args, 'bgp_best_path_selection_mode', None) is not None:
        network.routingConfig.bgpBestPathSelectionMode = messages.NetworkRoutingConfig.BgpBestPathSelectionModeValueValuesEnum(network_args.bgp_best_path_selection_mode)
    if getattr(network_args, 'bgp_bps_always_compare_med', None) is not None:
        network.routingConfig.bgpAlwaysCompareMed = network_args.bgp_bps_always_compare_med
    if getattr(network_args, 'bgp_bps_inter_region_cost', None) is not None:
        network.routingConfig.bgpInterRegionCost = messages.NetworkRoutingConfig.BgpInterRegionCostValueValuesEnum(network_args.bgp_bps_inter_region_cost)
    if hasattr(network_args, 'mtu') and network_args.mtu is not None:
        network.mtu = network_args.mtu
    if support_firewall_order and hasattr(network_args, 'network_firewall_policy_enforcement_order') and network_args.network_firewall_policy_enforcement_order:
        network.networkFirewallPolicyEnforcementOrder = messages.Network.NetworkFirewallPolicyEnforcementOrderValueValuesEnum(network_args.network_firewall_policy_enforcement_order)
    if hasattr(network_args, 'enable_ula_internal_ipv6'):
        network.enableUlaInternalIpv6 = network_args.enable_ula_internal_ipv6
    if hasattr(network_args, 'internal_ipv6_range'):
        network.internalIpv6Range = network_args.internal_ipv6_range
    if hasattr(network_args, 'rdma'):
        network.rdma = network_args.rdma
    return network