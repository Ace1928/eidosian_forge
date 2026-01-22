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
def AddEventFiltersPathPatternArg(parser, release_track, required=False, hidden=False):
    """Adds an argument for the trigger's event filters in path pattern format."""
    if release_track == base.ReleaseTrack.GA:
        parser.add_argument('--event-filters-path-pattern', action=arg_parsers.UpdateAction, type=arg_parsers.ArgDict(), hidden=hidden, required=required, help="The trigger's list of filters in path pattern format that apply to CloudEvent attributes. This flag can be repeated to add more filters to the list. Only events that match all these filters will be sent to the destination. Currently, path pattern format is only available for the resourceName attribute for Cloud Audit Log events.", metavar='ATTRIBUTE=PATH_PATTERN')