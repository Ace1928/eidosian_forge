from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddTopicResourceArg(parser, verb, positional=True, plural=False):
    """Add a resource argument for a Cloud Pub/Sub Topic.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    positional: bool, if True, means that the topic ID is a positional rather
      than a flag. If not positional, this also creates a '--topic-project' flag
      as subscriptions and topics do not need to be in the same project.
    plural: bool, if True, use a resource argument that returns a list.
  """
    concept_parsers.ConceptParser([CreateTopicResourceArg(verb, positional=positional, plural=plural)]).AddToParser(parser)