from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCertificateMapEntryResourceArg(parser, verb, noun=None, positional=True):
    """Add a resource argument for a Certificate Manager certificate map entry.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    noun: str, the resource; default: 'The certificate map'.
    positional: bool, if True, means that the map ID is a positional arg rather
      than a flag.
  """
    noun = noun or 'The certificate map entry'
    concept_parsers.ConceptParser([_GetCertificateMapEntryResourcePresentationSpec('entry' if positional else '--entry', noun, verb)]).AddToParser(parser)