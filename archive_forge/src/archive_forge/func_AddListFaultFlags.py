from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddListFaultFlags(parser):
    GetLocationResourceArg(required=True).AddToParser(parser)
    parser.add_argument('--service-name', type=str, help='service name.', required=False)
    parser.add_argument('--experiment-name', type=str, help='experiment name.', required=False)