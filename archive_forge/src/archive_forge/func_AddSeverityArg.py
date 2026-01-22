from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ids import ids_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddSeverityArg(parser, required=True, severity_levels=None):
    """Adds --severity flag."""
    choices = severity_levels or DEFAULT_SEVERITIES
    parser.add_argument('--severity', required=required, choices=choices, help='Minimum severity of threats to report on.')