from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddNetwork(parser):
    """Adds network argument for creating network endpoint groups."""
    help_text = '      Name of the network in which the NEG is created. `default` project\n      network is used if unspecified.\n  '
    network_applicable_ne_types = ['`gce-vm-ip-port`', '`non-gcp-private-ip-port`', '`gce-vm-ip`', '`private-service-connect`', '`internet-ip-port`', '`internet-fqdn-port`']
    help_text += '\n    This is only supported for NEGs with endpoint type {0}.\n\n    For Private Service Connect NEGs, you can optionally specify --network and\n    --subnet if --psc-target-service points to a published service. If\n    --psc-target-service points to the regional service endpoint of a Google\n    API, do not specify --network or --subnet.\n  '.format(_JoinWithOr(network_applicable_ne_types))
    parser.add_argument('--network', help=help_text)