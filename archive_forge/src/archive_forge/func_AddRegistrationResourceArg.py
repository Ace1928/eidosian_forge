from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddRegistrationResourceArg(parser, verb, noun=None, positional=True):
    """Add a resource argument for a Cloud Domains registration.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    noun: str, the resource; default: 'The domain registration'.
    positional: bool, if True, means that the registration ID is a positional
      arg rather than a flag.
  """
    noun = noun or 'The domain registration'
    concept_parsers.ConceptParser.ForResource('registration' if positional else '--registration', GetRegistrationResourceSpec(), '{} {}.'.format(noun, verb), required=True, flag_name_overrides={'location': ''}).AddToParser(parser)