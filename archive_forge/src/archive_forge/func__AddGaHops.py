from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.forwarding_rules import flags as ilb_flags
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.networks import flags as network_flags
from googlecloudsdk.command_lib.compute.routes import flags
from googlecloudsdk.command_lib.compute.vpn_tunnels import flags as vpn_flags
from googlecloudsdk.core import properties
def _AddGaHops(next_hop_group):
    """Attach arguments for GA next-hops to the a parser group."""
    next_hop_group.add_argument('--next-hop-instance', help="      Specifies the name of an instance that should handle traffic\n      matching this route. When this flag is specified, the zone of\n      the instance must be specified using\n      ``--next-hop-instance-zone''.\n      ")
    next_hop_group.add_argument('--next-hop-address', help="      Specifies the IP address of an instance that should handle\n      matching packets. The instance must have IP forwarding enabled\n      (i.e., include ``--can-ip-forward'' when creating the instance\n      using `gcloud compute instances create`)\n      ")
    flags.NEXT_HOP_GATEWAY_ARG.AddArgument(next_hop_group)
    next_hop_group.add_argument('--next-hop-vpn-tunnel', help='The target VPN tunnel that will receive forwarded traffic.')