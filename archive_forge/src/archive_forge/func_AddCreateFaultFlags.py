from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCreateFaultFlags(parser):
    GetFaultResourceArg().AddToParser(parser)
    parser.add_argument('--file', type=arg_parsers.YAMLFileContents(), metavar='FILE', help='The file containing the fault to be created.', required=True)