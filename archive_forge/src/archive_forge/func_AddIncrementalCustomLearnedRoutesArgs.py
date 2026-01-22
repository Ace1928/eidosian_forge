from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddIncrementalCustomLearnedRoutesArgs(parser):
    """Adds common arguments for incrementally updating custom learned routes.

  Args:
    parser: The parser to parse arguments.
  """
    incremental_args = parser.add_mutually_exclusive_group(required=False)
    incremental_args.add_argument('--add-custom-learned-route-ranges', type=arg_parsers.ArgList(), metavar='CIDR_RANGE', help='A list of user-defined custom learned route IP address ranges to\n              be added to this peer. This list is a comma separated IP address\n              ranges such as `1.2.3.4`,`6.7.0.0/16`,`2001:db8:abcd:12::/64`\n              where each IP address range must be a valid CIDR-formatted prefix.\n              If an IP address is provided without a subnet mask, it is\n              interpreted as a /32 singular IP address range for IPv4, and /128\n              for IPv6.')
    incremental_args.add_argument('--remove-custom-learned-route-ranges', type=arg_parsers.ArgList(), metavar='CIDR_RANGE', help='A list of user-defined custom learned route IP address ranges to\n              be removed from this peer. This list is a comma separated IP\n              address ranges such as `1.2.3.4`,`6.7.0.0/16`,`2001:db8:abcd:12::/64`\n              where each IP address range must be a valid CIDR-formatted prefix.\n              If an IP address is provided without a subnet mask, it is\n              interpreted as a /32 singular IP address range for IPv4, and /128\n              for IPv6.')