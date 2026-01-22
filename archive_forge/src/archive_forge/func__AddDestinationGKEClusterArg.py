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
def _AddDestinationGKEClusterArg(parser, required=False):
    """Adds an argument for the trigger's destination GKE service's cluster."""
    parser.add_argument('--destination-gke-cluster', required=required, help='Name of the GKE cluster that the destination GKE service is running in.  The cluster must be in the same project as the trigger.')