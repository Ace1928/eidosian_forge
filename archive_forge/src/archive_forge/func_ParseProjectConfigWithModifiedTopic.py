from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ParseProjectConfigWithModifiedTopic(args, project_config):
    """Parse and create a new ProjectConfig message with modified topic."""
    topic_name = GetTopicName(args)
    if args.add_topic:
        new_config = _ParsePubsubConfig(topic_name, args.message_format, args.service_account)
        return _AddTopicToResource(topic_name, new_config, project_config, resource_name='project')
    elif args.remove_topic:
        return _RemoveTopicFromResource(topic_name, project_config, resource_name='project')
    elif args.update_topic:
        return _UpdateTopicInResource(topic_name, args, project_config, resource_name='project')
    return project_config