from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddNoteTemplateFlagToParser(parser, required=True):
    parser.add_argument('--node-template', required=required, help='The name of the node template resource to be set for this node group.')