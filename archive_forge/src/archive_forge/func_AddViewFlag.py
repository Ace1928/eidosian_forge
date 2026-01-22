from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.network_connectivity import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddViewFlag(parser):
    """Add the --view argument to the given parser."""
    view_help_text = '        Enumeration to control which spoke fields are included in the response.'
    parser.add_argument('--view', required=False, choices=['basic', 'detailed'], default='basic', help=view_help_text)