from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def ApplyTopicsUpdate(args, original_topics):
    """Applies updates to the list of topics on a secret.

  Preserves the original order of topics.

  Args:
    args (argparse.Namespace): The collection of user-provided arguments.
    original_topics (list): Topics configured on the secret prior to update.

  Returns:
      result (list): List of strings of topic names after update.
  """
    if args.IsSpecified('clear_topics'):
        return []
    topics_set = set()
    for topic in original_topics:
        topics_set.add(topic.name)
    if args.IsSpecified('remove_topics'):
        for topic_name in args.remove_topics:
            topics_set.discard(topic_name)
        new_topics = []
        for topic in original_topics:
            if topic.name in topics_set:
                new_topics.append(topic.name)
        return new_topics
    if args.IsSpecified('add_topics'):
        new_topics = []
        for topic in original_topics:
            new_topics.append(topic.name)
        for topic_name in args.add_topics:
            if topic_name not in topics_set:
                new_topics.append(topic_name)
        return new_topics