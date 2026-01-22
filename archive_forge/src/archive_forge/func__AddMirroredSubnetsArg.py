from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddMirroredSubnetsArg(parser, is_for_update=False):
    """Adds args to specify mirrored subnets."""
    if is_for_update:
        subnets = parser.add_mutually_exclusive_group(help='      Update the mirrored subnets of this packet mirroring.\n      ')
        subnets.add_argument('--add-mirrored-subnets', type=arg_parsers.ArgList(), metavar='SUBNET', help='List of subnets to add to the packet mirroring.')
        subnets.add_argument('--remove-mirrored-subnets', type=arg_parsers.ArgList(), metavar='SUBNET', help='List of subnets to remove from the packet mirroring.')
        subnets.add_argument('--set-mirrored-subnets', type=arg_parsers.ArgList(), metavar='SUBNET', help='List of subnets to be mirrored on the packet mirroring.')
        subnets.add_argument('--clear-mirrored-subnets', action='store_true', default=None, help='If specified, clear the existing subnets from the packet mirroring.')
    else:
        parser.add_argument('--mirrored-subnets', type=arg_parsers.ArgList(), metavar='SUBNET', help='        List of subnets to be mirrored.\n        You can provide this as the full URL to the subnet, partial URL, or\n        name.\n        For example, the following are valid values:\n          * https://compute.googleapis.com/compute/v1/projects/myproject/\n            regions/us-central1/subnetworks/subnet-1\n          * projects/myproject/regions/us-central1/subnetworks/subnet-1\n          * subnet-1\n        ')