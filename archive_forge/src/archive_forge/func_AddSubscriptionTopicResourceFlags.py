from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def AddSubscriptionTopicResourceFlags(parser):
    """Adds --topic and --topic-project flags to a parser."""
    parser.add_argument('--topic', required=True, help='The name of the topic from which this subscription is receiving messages. Each subscription is attached to a single topic.')
    parser.add_argument('--topic-project', help='The name of the project the provided topic belongs to. If not set, it defaults to the currently selected cloud project.')