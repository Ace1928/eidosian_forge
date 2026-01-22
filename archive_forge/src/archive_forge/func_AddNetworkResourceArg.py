from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddNetworkResourceArg(parser, verb, positional=False):
    """Add a resource argument for a GDCE network.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to create'.
    positional: bool, if True, means that the resource is a positional rather
      than a flag.
  """
    if positional:
        name = 'network'
    else:
        name = '--network'
    resource_specs = [presentation_specs.ResourcePresentationSpec(name, GetNetworkResourceSpec(), 'The network {}.'.format(verb), required=True)]
    concept_parsers.ConceptParser(resource_specs).AddToParser(parser)