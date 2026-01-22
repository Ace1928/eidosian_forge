from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import log
@base.Deprecate(is_removed=False, warning='This command is deprecated. Use `gcloud beta tasks queues create` instead')
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class CreateAppEngine(base.CreateCommand):
    """Create a Cloud Tasks queue.

  The flags available to this command represent the fields of a queue that are
  mutable.
  """
    detailed_help = {'DESCRIPTION': '          {description}\n          ', 'EXAMPLES': '          To create a Cloud Tasks queue:\n\n              $ {command} my-queue\n                --max-attempts=10 --max-retry-duration=5s\n                --max-doublings=4 --min-backoff=1s\n                --max-backoff=10s\n                --max-dispatches-per-second=100\n                --max-concurrent-dispatches=10\n                --routing-override=service:abc\n         '}

    def __init__(self, *args, **kwargs):
        super(CreateAppEngine, self).__init__(*args, **kwargs)
        self.is_alpha = False

    @staticmethod
    def Args(parser):
        flags.AddQueueResourceArg(parser, 'to create')
        flags.AddLocationFlag(parser)
        flags.AddCreatePushQueueFlags(parser, release_track=base.ReleaseTrack.BETA, app_engine_queue=True, http_queue=False)

    def Run(self, args):
        api = GetApiAdapter(self.ReleaseTrack())
        queues_client = api.queues
        queue_ref = parsers.ParseQueue(args.queue, args.location)
        location_ref = parsers.ExtractLocationRefFromQueueRef(queue_ref)
        queue_config = parsers.ParseCreateOrUpdateQueueArgs(args, constants.PUSH_QUEUE, api.messages, release_track=self.ReleaseTrack(), http_queue=False)
        if not self.is_alpha:
            create_response = queues_client.Create(location_ref, queue_ref, retry_config=queue_config.retryConfig, rate_limits=queue_config.rateLimits, app_engine_http_queue=queue_config.appEngineHttpQueue, stackdriver_logging_config=queue_config.stackdriverLoggingConfig)
        else:
            create_response = queues_client.Create(location_ref, queue_ref, retry_config=queue_config.retryConfig, rate_limits=queue_config.rateLimits, app_engine_http_target=queue_config.appEngineHttpTarget)
        log.CreatedResource(parsers.GetConsolePromptString(queue_ref.RelativeName()), 'queue')
        return create_response