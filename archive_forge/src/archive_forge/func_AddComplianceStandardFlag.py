from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.audit_manager import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddComplianceStandardFlag(parser, required=True):
    parser.add_argument('--compliance-standard', help='Compliance Standard against which the Report must be generated. Eg: FEDRAMP_MODERATE', required=required)