from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseSourceRepoWithModifiedTopic(args, repo):
    """Parse and create a new Repo message with modified topic."""
    topic_name = GetTopicName(args)
    if args.add_topic:
        new_config = _ParsePubsubConfig(topic_name, args.message_format, args.service_account)
        return _AddTopicToResource(topic_name, new_config, repo, resource_name='repo')
    elif args.remove_topic:
        return _RemoveTopicFromResource(topic_name, repo, resource_name='repo')
    elif args.update_topic:
        return _UpdateTopicInResource(topic_name, args, repo, resource_name='repo')
    return repo