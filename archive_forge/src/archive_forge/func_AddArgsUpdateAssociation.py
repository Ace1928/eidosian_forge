from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddArgsUpdateAssociation(parser):
    """Adds the arguments of association update."""
    parser.add_argument('--name', required=True, help='Name of the association.')
    parser.add_argument('--priority', required=True, help='Priority of the association.')