from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddIpVersionGroup(parser):
    """Adds IP versions flag in a mutually exclusive group."""
    parser.add_argument('--ip-version', choices=['IPV4', 'IPV6'], type=lambda x: x.upper(), help='      Version of the IP address to be allocated and reserved.\n      The default is IPV4.\n\n      IP version can only be specified for global addresses that are generated\n      automatically (i.e., along with\n      the `--global` flag, given `--addresses` is not specified) and if the\n      `--network-tier` is `PREMIUM`.\n      ')