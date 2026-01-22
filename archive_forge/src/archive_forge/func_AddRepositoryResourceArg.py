from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddRepositoryResourceArg(parser, verb):
    """Add a resource argument for a Secure Source Manager repository.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
  """
    concept_parsers.ConceptParser.ForResource('repository', GetRepositoryResourceSpec(), 'The Secure Source Manager repository {}.'.format(verb), required=True).AddToParser(parser)