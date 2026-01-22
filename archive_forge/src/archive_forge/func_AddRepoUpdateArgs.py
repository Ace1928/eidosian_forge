from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.source import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddRepoUpdateArgs(parser, verb='to update'):
    """Add the arg groups for `source repos update` command."""
    topic_group = parser.add_group('Manages Cloud Pub/Sub topics associated with the repository.', required=True)
    topic_resource_group = topic_group.add_argument_group(mutex=True, required=True)
    concept_parsers.ConceptParser([resource_args.CreateTopicResourcePresentationSpec('add', 'The Cloud Pub/Sub topic to add to the repository.', topic_resource_group), resource_args.CreateTopicResourcePresentationSpec('remove', 'The Cloud Pub/Sub topic to remove from the repository.', topic_resource_group), resource_args.CreateTopicResourcePresentationSpec('update', 'The Cloud Pub/Sub topic to update in the project.', topic_resource_group), resource_args.CreateRepoResourcePresentationSpec(verb)], command_level_fallthroughs={'--add-topic.project': ['--topic-project'], '--remove-topic.project': ['--topic-project'], '--update-topic.project': ['--topic-project']}).AddToParser(parser)
    AddOptionalTopicFlags(topic_group)