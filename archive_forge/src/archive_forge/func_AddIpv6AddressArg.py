from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddIpv6AddressArg(parser):
    parser.add_argument('--ipv6-address', type=str, help='\n        Assigns the given external IPv6 address to an instance.\n        The address must be the first IP in the range. This option is applicable\n        only to dual-stack instances with stack-type=IPV4_ONLY.\n      ')