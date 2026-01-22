from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import arg_parsers
def AddBgpPeerArgs(parser, for_update=False, enable_peer_ipv6_range=False):
    """Adds common arguments for managing BGP peers."""
    operation = 'added'
    if for_update:
        operation = 'updated'
    parser.add_argument('--interface', required=not for_update, help='The name of the interface for this BGP peer.')
    parser.add_argument('--peer-name', required=True, help='The name of the new BGP peer being {0}.'.format(operation))
    parser.add_argument('--peer-asn', required=not for_update, type=int, help='The BGP autonomous system number (ASN) for this BGP peer. Must be a 16-bit or 32-bit private ASN as defined in https://tools.ietf.org/html/rfc6996, for example `--asn=64512`.')
    ip_address_parser = parser.add_mutually_exclusive_group(required=not for_update)
    ip_address_parser.add_argument('--peer-ipv4-range', help='The IPv4 link-local address range of the peer router.')
    if enable_peer_ipv6_range:
        ip_address_parser.add_argument('--peer-ipv6-range', help='The IPv6 link-local address range of the peer router.')