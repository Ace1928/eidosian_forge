from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddInternalIpv6PrefixLengthArg(parser):
    parser.add_argument('--internal-ipv6-prefix-length', type=int, help='\n        Optional field that indicates the prefix length of the internal IPv6\n        address range, should be used together with\n        `--internal-ipv6-address=fd20::`. Only /96 IP address range is supported\n        and the default value is 96. If not set, then  either the prefix length\n        from `--internal-ipv6-address=fd20::/96` will be used or the default\n        value of 96 will be assigned.\n      ')