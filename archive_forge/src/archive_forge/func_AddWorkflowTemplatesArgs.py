from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.args import labels_util
def AddWorkflowTemplatesArgs(parser, api_version):
    """Register flags for this command."""
    labels_util.AddCreateLabelsFlags(parser)
    flags.AddTemplateResourceArg(parser, 'add job to', api_version, positional=False)
    parser.add_argument('--step-id', required=True, type=str, help='The step ID of the job in the workflow template.')
    parser.add_argument('--start-after', metavar='STEP_ID', type=arg_parsers.ArgList(element_type=str, min_length=1), help='(Optional) List of step IDs to start this job after.')