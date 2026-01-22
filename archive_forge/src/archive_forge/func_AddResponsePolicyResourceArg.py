from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddResponsePolicyResourceArg(parser, verb, api_version, positional=True, required=True):
    """Add a resource argument for a Cloud DNS Response Policy.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    api_version: str, the version of the API to use.
    positional: bool, if True, means that the policy name is a positional rather
      than a flag.
    required: bool, if True, means that the arg will be required.
  """
    if positional:
        name = 'response_policies'
    else:
        name = '--response_policies'
    concept_parsers.ConceptParser.ForResource(name, GetResponsePolicyResourceSpec(api_version), 'The response policy {}.'.format(verb), required=required).AddToParser(parser)