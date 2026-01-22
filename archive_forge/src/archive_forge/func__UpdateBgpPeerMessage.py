from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.routers import flags
from googlecloudsdk.command_lib.compute.routers import router_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def _UpdateBgpPeerMessage(peer, messages, args, md5_authentication_key_name, support_bfd_mode=False, enable_ipv6_bgp=False, enable_route_policies=False):
    """Updates base attributes of a BGP peer based on flag arguments."""
    attrs = {'interfaceName': args.interface, 'ipAddress': args.ip_address, 'peerIpAddress': args.peer_ip_address, 'peerAsn': args.peer_asn, 'advertisedRoutePriority': args.advertised_route_priority}
    if args.enabled is not None:
        if args.enabled:
            attrs['enable'] = messages.RouterBgpPeer.EnableValueValuesEnum.TRUE
        else:
            attrs['enable'] = messages.RouterBgpPeer.EnableValueValuesEnum.FALSE
    if args.enable_ipv6 is not None:
        attrs['enableIpv6'] = args.enable_ipv6
    if args.ipv6_nexthop_address is not None:
        attrs['ipv6NexthopAddress'] = args.ipv6_nexthop_address
    if args.peer_ipv6_nexthop_address is not None:
        attrs['peerIpv6NexthopAddress'] = args.peer_ipv6_nexthop_address
    if enable_ipv6_bgp and args.enable_ipv4 is not None:
        attrs['enableIpv4'] = args.enable_ipv4
    if enable_ipv6_bgp and args.ipv4_nexthop_address is not None:
        attrs['ipv4NexthopAddress'] = args.ipv4_nexthop_address
    if enable_ipv6_bgp and args.peer_ipv4_nexthop_address is not None:
        attrs['peerIpv4NexthopAddress'] = args.peer_ipv4_nexthop_address
    if args.custom_learned_route_priority is not None:
        attrs['customLearnedRoutePriority'] = args.custom_learned_route_priority
    if args.md5_authentication_key is not None:
        attrs['md5AuthenticationKeyName'] = md5_authentication_key_name
    if enable_route_policies:
        attrs['exportPolicies'] = args.export_policies
        attrs['importPolicies'] = args.import_policies
    for attr, value in attrs.items():
        if value is not None:
            setattr(peer, attr, value)
    if args.clear_md5_authentication_key:
        peer.md5AuthenticationKeyName = None
    if support_bfd_mode:
        bfd = _UpdateBgpPeerBfdMessageMode(messages, peer, args)
    else:
        bfd = _UpdateBgpPeerBfdMessage(messages, peer, args)
    if bfd is not None:
        setattr(peer, 'bfd', bfd)