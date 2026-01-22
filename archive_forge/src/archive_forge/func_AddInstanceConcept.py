from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddInstanceConcept(parser, instance_concept_str):
    concept_parsers.ConceptParser([GetInstancePresentationSpec(instance_concept_str)]).AddToParser(parser)