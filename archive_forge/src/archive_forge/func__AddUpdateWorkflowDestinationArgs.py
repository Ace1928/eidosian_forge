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
def _AddUpdateWorkflowDestinationArgs(parser, required=False, hidden=False):
    """Adds arguments related to trigger's Workflow destination for update operations."""
    workflow_group = parser.add_group(required=required, hidden=hidden, help='Flags for updating a Workflow destination.')
    _AddDestinationWorkflowArg(workflow_group)
    _AddDestinationWorkflowLocationArg(workflow_group)