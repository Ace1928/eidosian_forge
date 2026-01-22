from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCertificateResourceArg(parser, verb, noun=None, name='certificate', positional=True, required=True, plural=False, group=None, with_location=True):
    """Add a resource argument for a Certificate Manager certificate.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    noun: str, the resource; default: 'The certificate'.
    name: str, the name of the flag.
    positional: bool, if True, means that the certificate ID is a positional arg
      rather than a flag.
    required: bool, if True the flag is required.
    plural: bool, if True the flag is a list.
    group: args group.
    with_location: bool, if False, means that location flag is hidden.
  """
    noun = noun or 'The certificate'
    concept_parsers.ConceptParser([_GetCertificateResourcePresentationSpec('certificate' if positional else '--' + name, noun, verb, required, plural, group, with_location)]).AddToParser(parser)