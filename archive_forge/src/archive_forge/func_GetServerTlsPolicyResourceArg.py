from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetServerTlsPolicyResourceArg(verb, noun='The server TLS policy', name='server-tls-policy', required=False, plural=False, group=None, region_fallthrough=False):
    """Creates a resource argument for a Server TLS policy.

  Args:
    verb: str, the verb to describe the resource, such as 'to update'.
    noun: str, the resource; default: 'The server TLS policy'.
    name: str, the name of the flag.
    required: bool, if True the flag is required.
    plural: bool, if True the flag is a list.
    group: args group.
    region_fallthrough: bool, True if the command has a region flag that should
      be used as a fallthrough for the server TLS policy location.

  Returns:
    ServerTlsPolicyResourceArg: ConceptParser, holder for Server TLS policy
    argument.
  """
    return concept_parsers.ConceptParser([_GetServerTlsPolicyResourcePresentationSpec('--' + name, noun, verb, required, plural, group, region_fallthrough)])