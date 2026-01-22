from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddArgsUpdateSp(parser):
    """Adds the argument for security policy update."""
    parser.add_argument('--organization', help='Organization in which the organization security policy is to be updated. Must be set if SECURITY_POLICY is display name.')
    parser.add_argument('--description', help='An optional, textual description for the organization security policy.')