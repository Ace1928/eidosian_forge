from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import log
@base.Deprecate(is_removed=False, warning='This command is deprecated. Use `gcloud alpha tasks queues update` instead')
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AlphaUpdateAppEngine(UpdateAppEngine):
    """Update a Cloud Tasks queue.

  The flags available to this command represent the fields of a queue that are
  mutable. Attempting to use this command on a different type of queue will
  result in an error.
  """
    detailed_help = {'DESCRIPTION': '          {description}\n          ', 'EXAMPLES': '          To update a Cloud Tasks queue:\n\n              $ {command} my-queue\n                --clear-max-attempts --clear-max-retry-duration\n                --clear-max-doublings --clear-min-backoff\n                --clear-max-backoff\n                --clear-max-tasks-dispatched-per-second\n                --clear-max-concurrent-tasks\n                --clear-routing-override\n         '}

    def __init__(self, *args, **kwargs):
        super(AlphaUpdateAppEngine, self).__init__(*args, **kwargs)
        self.is_alpha = True

    @staticmethod
    def Args(parser):
        flags.AddQueueResourceArg(parser, 'to update')
        flags.AddLocationFlag(parser)
        flags.AddUpdatePushQueueFlags(parser, release_track=base.ReleaseTrack.ALPHA, app_engine_queue=True, http_queue=False)