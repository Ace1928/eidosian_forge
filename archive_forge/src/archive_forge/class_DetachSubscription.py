from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_ex
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class DetachSubscription(base.UpdateCommand):
    """Detaches one or more Cloud Pub/Sub subscriptions."""
    detailed_help = {'EXAMPLES': '          To detach a Cloud Pub/Sub subscription, run:\n\n              $ {command} mysubscription'}

    @staticmethod
    def Args(parser):
        resource_args.AddSubscriptionResourceArg(parser, 'to detach.', plural=True)

    def Run(self, args):
        return _Run(args)