from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddCandidateSubnets(parser):
    """Adds candidate subnets flag to the argparse.ArgumentParser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('--candidate-subnets', type=arg_parsers.ArgList(max_length=16), metavar='SUBNET', help="      Up to 16 candidate prefixes that can be used to restrict the allocation of\n      `cloudRouterIpAddress` and `customerRouterIpAddress` for this\n      attachment. All prefixes must be within link-local address space.\n      Google attempts to select an unused subnet of SUBNET_LENGTH from the\n      supplied candidate subnet(s), or all of link-local space if no subnets\n      supplied. Google does not re-use a subnet already in-use by your project,\n      even if it's contained in one of the candidate subnets. The request fails\n      if all candidate subnets are in use at Google's edge.", default=[])