from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListSubscriptions(base.ListCommand):
    """Lists Cloud Pub/Sub subscriptions from a given topic."""
    detailed_help = {'DESCRIPTION': '          Lists all of the Cloud Pub/Sub subscriptions attached to the given\n          topic and that match the given filter.', 'EXAMPLES': '          To filter results by subscription name\n          (ie. only show subscription \'mysubs\'), run:\n\n            $ {command} mytopic --filter=mysubs\n\n          To combine multiple filters (with AND or OR), run:\n\n            $ {command} mytopic --filter="mysubs1 OR mysubs2"\n          '}

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('yaml')
        parser.display_info.AddUriFunc(util.SubscriptionUriFunc)
        resource_args.AddTopicResourceArg(parser, 'to list subscriptions for.')

    def Run(self, args):
        return _Run(args)