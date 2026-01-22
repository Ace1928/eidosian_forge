from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def CreateConditionResourceArg(verb):
    help_text = 'The name of the Condition to {}.'.format(verb)
    return presentation_specs.ResourcePresentationSpec('condition', GetConditionResourceSpec(), help_text, required=True, prefixes=False)