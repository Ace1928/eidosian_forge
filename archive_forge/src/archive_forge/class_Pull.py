from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_ex
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.api_lib.util import exceptions as util_ex
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.pubsub import flags
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
import six
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Pull(base.ListCommand):
    """Pulls one or more Cloud Pub/Sub messages from a subscription."""
    detailed_help = {'DESCRIPTION': '          Returns one or more messages from the specified Cloud Pub/Sub\n          subscription, if there are any messages enqueued.\n\n          By default, this command returns only one message from the\n          subscription. Use the `--limit` flag to specify the max messages to\n          return.\n\n          Please note that this command is not guaranteed to return all the\n          messages in your backlog or the maximum specified in the --limit\n          argument.  Receiving fewer messages than available occasionally\n          is normal.'}

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat(MESSAGE_FORMAT_WITH_ACK_STATUS)
        resource_args.AddSubscriptionResourceArg(parser, 'to pull messages from.')
        flags.AddPullFlags(parser)
        base.LIMIT_FLAG.SetDefault(parser, 1)

    def Run(self, args):
        return _Run(args, args.limit)