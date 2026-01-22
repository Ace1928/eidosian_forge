from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.base import ReleaseTrack
from googlecloudsdk.command_lib.assured import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddListWorkloadsFlags(parser):
    parser.add_argument('--location', required=True, help='The location of the Assured Workloads environments. For a current list of supported LOCATION values, see [Assured Workloads locations](https://cloud.google.com/assured-workloads/docs/locations).')
    parser.add_argument('--organization', required=True, help='The parent organization of the Assured Workloads environments, provided as an organization ID.')