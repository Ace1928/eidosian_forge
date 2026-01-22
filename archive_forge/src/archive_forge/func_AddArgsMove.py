from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddArgsMove(parser):
    """Adds the argument for firewall policy move."""
    parser.add_argument('--organization', help='Organization in which the organization firewall policy is to be moved. Must be set if FIREWALL_POLICY is short name.')
    parser.add_argument('--folder', help='Folder to which the organization firewall policy is to be moved.')