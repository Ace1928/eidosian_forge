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
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class PullBeta(Pull):
    """Pulls one or more Cloud Pub/Sub messages from a subscription."""

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat(MESSAGE_FORMAT_WITH_ACK_STATUS)
        resource_args.AddSubscriptionResourceArg(parser, 'to pull messages from.')
        flags.AddPullFlags(parser, add_deprecated=True, add_wait=True, add_return_immediately=True)

    def Run(self, args):
        if args.IsSpecified('limit'):
            if args.IsSpecified('max_messages'):
                raise exceptions.ConflictingArgumentsException('--max-messages', '--limit')
            max_messages = args.limit
        else:
            max_messages = args.max_messages
        return_immediately = False
        if args.IsSpecified('return_immediately'):
            return_immediately = args.return_immediately
        elif args.IsSpecified('wait'):
            return_immediately = not args.wait
        return _Run(args, max_messages, return_immediately)