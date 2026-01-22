from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddGatewayResourceArg(parser, verb, positional=False, required=True):
    """Adds Gateway resource argument to parser.

  Args:
    parser: parser to add arg to
    verb: action being taken with the Gateway
    positional: Boolean indicating if argument is positional, default False
    required: Boolean for if this is required, default is True

  Returns: None
  """
    if positional:
        name = 'gateway'
    else:
        name = '--gateway'
    concept_parsers.ConceptParser.ForResource(name, GetGatewayResourceSpec(), 'Name for gateway which will be {}.'.format(verb), required=required).AddToParser(parser)