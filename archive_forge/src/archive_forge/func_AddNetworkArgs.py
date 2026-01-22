from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddNetworkArgs(parser):
    """Set arguments for choosing the network/subnetwork."""
    parser.add_argument('--network', help='      Specifies the network for the VMs that are created from the imported\n      machine image. If `--subnet` is also specified, then the subnet must\n      be a subnetwork of network specified by `--network`. If neither is\n      specified, the `default` network is used.\n      ')
    parser.add_argument('--subnet', help='      Specifies the subnet for the VMs created from the imported machine\n      image. If `--network` is also specified, the subnet must be\n      a subnetwork of the network specified by `--network`.\n      ')