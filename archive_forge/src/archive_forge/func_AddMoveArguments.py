from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util import completers
def AddMoveArguments(parser):
    """Add flags for move."""
    parser.add_argument('--target-project', required=True, help='The target project to move address to. It can be either a project name or a project numerical ID. It must not be the same as the current project.')
    parser.add_argument('--new-name', help="Name of moved new address. If not specified, current address's name is used.")
    parser.add_argument('--description', help='Description of moved new address.')