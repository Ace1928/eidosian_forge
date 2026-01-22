from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.immersive_stream.xr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddContentResourceArg(parser, verb, positional=True):
    """Adds a resource argument for an Immersive Stream for XR content resource.

  Args:
    parser: The argparse parser to add the resource arg to.
    verb: str, the verb to describe the resource, such as 'to update'.
    positional: bool, whether the argument is positional or not.
  """
    name = 'content' if positional else '--content'
    concept_parsers.ConceptParser.ForResource(name, GetContentResourceSpec(), 'Immersive Stream for XR content resource {}.'.format(verb), required=True).AddToParser(parser)