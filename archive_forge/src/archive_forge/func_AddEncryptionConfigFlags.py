from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddEncryptionConfigFlags(parser, verb):
    """Add a resource argument for a KMS Key used to create a CMEK encrypted resource.

  Args:
    parser: argparser, the parser for the command.
    verb: str, the verb used to describe the resource, such as 'to create'.
  """
    concept_parsers.ConceptParser.ForResource('--kms-key', GetKmsKeyResourceSpec(), 'Cloud KMS key to be used {}.'.format(verb), required=False).AddToParser(parser)