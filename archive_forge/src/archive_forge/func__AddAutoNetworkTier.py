from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def _AddAutoNetworkTier(parser):
    choices = {'PREMIUM': 'High quality, Google-grade network tier with support for all networking products.', 'STANDARD': 'Public internet quality, with only limited support for other networking products.'}
    parser.add_argument('--auto-network-tier', help=textwrap.dedent('The network tier to use when automatically reserving NAT IP addresses.'), choices=choices, required=False)