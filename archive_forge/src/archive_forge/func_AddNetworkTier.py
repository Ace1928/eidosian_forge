from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddNetworkTier(parser):
    """Adds network tier flag."""
    parser.add_argument('--network-tier', type=lambda x: x.upper(), help="      The network tier to assign to the reserved IP addresses. ``NETWORK_TIER''\n      must be one of: `PREMIUM`, `STANDARD`, `FIXED_STANDARD`.\n      The default value is `PREMIUM`.\n\n      While regional external addresses (`--region` specified, `--subnet`\n      omitted) can use either `PREMIUM` or `STANDARD`, global external\n      addresses (`--global` specified, `--subnet` omitted) can only use\n      `PREMIUM`. Internal addresses can only use `PREMIUM`.\n      ")