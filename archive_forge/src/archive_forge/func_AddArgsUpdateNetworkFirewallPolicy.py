from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddArgsUpdateNetworkFirewallPolicy(parser):
    """Adds the arguments  for firewall policy update."""
    parser.add_argument('--description', help='An optional, textual description for the network firewall policy.')