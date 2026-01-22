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
def AddAckDeadlineFlag(parser, required=False):
    parser.add_argument('--ack-deadline', type=int, required=required, help='The number of seconds the system will wait for a subscriber to acknowledge receiving a message before re-attempting delivery.')