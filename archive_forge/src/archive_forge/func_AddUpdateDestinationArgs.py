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
def AddUpdateDestinationArgs(parser, release_track, required=False):
    """Adds arguments related to trigger's destination for update operations."""
    dest_group = parser.add_mutually_exclusive_group(required=required, help='Flags for updating the destination to which events should be sent.')
    _AddUpdateCloudRunDestinationArgs(dest_group, release_track)
    if release_track == base.ReleaseTrack.GA:
        _AddUpdateGKEDestinationArgs(dest_group)
        _AddUpdateWorkflowDestinationArgs(dest_group, hidden=True)
        _AddUpdateFunctionDestinationArgs(dest_group, hidden=True)