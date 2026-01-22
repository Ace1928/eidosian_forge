from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apphub import utils as apphub_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCreateApplicationWorkloadFlags(parser, release_track=base.ReleaseTrack.ALPHA):
    """Adds flags required to create an application workload."""
    concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('WORKLOAD', GetApplicationWorkloadResourceSpec(), 'The Workload resource.', flag_name_overrides={'location': '--location', 'application': '--application'}, prefixes=True, required=True), presentation_specs.ResourcePresentationSpec('--discovered-workload', GetDiscoveredWorkloadResourceSpec(), 'The discovered workload resource.', flag_name_overrides={'location': '', 'project': ''}, prefixes=True, required=True)], command_level_fallthroughs={'--discovered-workload.location': ['WORKLOAD.location']}).AddToParser(parser)
    AddAttributesFlags(parser, resource_name='workload', release_track=release_track)
    parser.add_argument('--display-name', help='Human-friendly display name')
    parser.add_argument('--description', help='Description of the Workload')
    parser.add_argument('--async', dest='async_', action='store_true', default=False, help='Return immediately, without waiting for the operation in progress to complete.')