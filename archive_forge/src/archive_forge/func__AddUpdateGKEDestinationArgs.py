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
def _AddUpdateGKEDestinationArgs(parser, required=False, hidden=False):
    """Adds arguments related to trigger's GKE service destination for update operations."""
    gke_group = parser.add_group(required=required, hidden=hidden, help='Flags for updating a GKE service destination.')
    _AddDestinationGKENamespaceArg(gke_group)
    _AddDestinationGKEServiceArg(gke_group)
    destination_gke_path_group = gke_group.add_mutually_exclusive_group()
    _AddDestinationGKEPathArg(destination_gke_path_group)
    _AddClearDestinationGKEPathArg(destination_gke_path_group)