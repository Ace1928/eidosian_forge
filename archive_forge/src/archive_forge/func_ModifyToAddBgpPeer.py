from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from apitools.base.py import encoding
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
import six
def ModifyToAddBgpPeer(self, args, existing):
    """Mutate the router to add a BGP peer."""
    replacement = encoding.CopyProtoMessage(existing)
    bgp_peer_args = {'name': args.peer_name, 'interface': args.interface, 'peerAsn': args.peer_asn}
    if args.peer_ipv4_range is not None:
        bgp_peer_args['peerIpv4Cidr'] = args.peer_ipv4_range
    if 'peer_ipv6_range' in args and args.peer_ipv6_range is not None:
        bgp_peer_args['peerIpv6Cidr'] = args.peer_ipv6_range
    new_bgp_peer = self._messages.BgpPeer(**bgp_peer_args)
    replacement.bgpPeer.append(new_bgp_peer)
    return replacement