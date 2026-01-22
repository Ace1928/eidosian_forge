from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddExportInstanceArgs(parser):
    """Register flags Export Instance command."""
    AddInstanceConcept(parser, 'Arguments and flags that specify the Looker instance you want to export.')
    AddTargetGcsUriGroup(parser)
    AddKmsKeyGroup(parser)