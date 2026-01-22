from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import googlecloudsdk
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def _AddDestinationWorkflowArg(parser, required=False):
    """Adds an argument for the trigger's destination Workflow."""
    parser.add_argument('--destination-workflow', required=required, help='ID of the Workflow that receives the events for the trigger. The Workflow must be in the same project as the trigger.')