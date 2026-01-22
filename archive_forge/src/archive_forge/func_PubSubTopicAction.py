from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def PubSubTopicAction(topic):
    """Return PubSub DlpV2Action for given PubSub topic."""
    action_msg = _GetMessageClass('GooglePrivacyDlpV2Action')
    pubsub_action = _GetMessageClass('GooglePrivacyDlpV2PublishToPubSub')
    return action_msg(pubSub=pubsub_action(topic=topic))