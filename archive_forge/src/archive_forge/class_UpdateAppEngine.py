from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.tasks import GetApiAdapter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import log
@base.Deprecate(is_removed=False, warning='This command is deprecated. Use `gcloud beta tasks queues update` instead')
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class UpdateAppEngine(base.UpdateCommand):
    """Update a Cloud Tasks queue.

  The flags available to this command represent the fields of a queue that are
  mutable.
  """
    detailed_help = {'DESCRIPTION': '          {description}\n          ', 'EXAMPLES': '          To update a Cloud Tasks queue:\n\n              $ {command} my-queue\n                --clear-max-attempts --clear-max-retry-duration\n                --clear-max-doublings --clear-min-backoff\n                --clear-max-backoff\n                --clear-max-dispatches-per-second\n                --clear-max-concurrent-dispatches\n                --clear-routing-override\n         '}

    def __init__(self, *args, **kwargs):
        super(UpdateAppEngine, self).__init__(*args, **kwargs)
        self.is_alpha = False

    @staticmethod
    def Args(parser):
        flags.AddQueueResourceArg(parser, 'to update')
        flags.AddLocationFlag(parser)
        flags.AddUpdatePushQueueFlags(parser, release_track=base.ReleaseTrack.BETA, app_engine_queue=True, http_queue=False)

    def Run(self, args):
        parsers.CheckUpdateArgsSpecified(args, constants.PUSH_QUEUE, release_track=self.ReleaseTrack())
        api = GetApiAdapter(self.ReleaseTrack())
        queues_client = api.queues
        queue_ref = parsers.ParseQueue(args.queue, args.location)
        queue_config = parsers.ParseCreateOrUpdateQueueArgs(args, constants.PUSH_QUEUE, api.messages, is_update=True, release_track=self.ReleaseTrack(), http_queue=False)
        updated_fields = parsers.GetSpecifiedFieldsMask(args, constants.PUSH_QUEUE, release_track=self.ReleaseTrack())
        if not self.is_alpha:
            app_engine_routing_override = queue_config.appEngineHttpQueue.appEngineRoutingOverride if queue_config.appEngineHttpQueue is not None else None
            update_response = queues_client.Patch(queue_ref, updated_fields, retry_config=queue_config.retryConfig, rate_limits=queue_config.rateLimits, app_engine_routing_override=app_engine_routing_override, stackdriver_logging_config=queue_config.stackdriverLoggingConfig)
        else:
            app_engine_routing_override = queue_config.appEngineHttpTarget.appEngineRoutingOverride if queue_config.appEngineHttpTarget is not None else None
            update_response = queues_client.Patch(queue_ref, updated_fields, retry_config=queue_config.retryConfig, rate_limits=queue_config.rateLimits, app_engine_routing_override=app_engine_routing_override)
        log.status.Print('Updated queue [{}].'.format(parsers.GetConsolePromptString(queue_ref.RelativeName())))
        return update_response