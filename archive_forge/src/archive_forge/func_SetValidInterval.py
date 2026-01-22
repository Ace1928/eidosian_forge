from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as sdk_core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def SetValidInterval(api_version):
    """Combine multiple timing constraints into a valid_interval."""

    def Process(ref, args, request):
        del ref
        if args.valid_after_duration and args.valid_after_time or (args.valid_until_duration and args.valid_until_time):
            raise exceptions.ConflictingArgumentsException('Only one timing constraint for each of (start, end) time is permitted')
        tpu_messages = GetMessagesModule(api_version)
        current_time = times.Now()
        start_time = None
        if args.valid_after_time:
            start_time = args.valid_after_time
        elif args.valid_after_duration:
            start_time = args.valid_after_duration.GetRelativeDateTime(current_time)
        end_time = None
        if args.valid_until_time:
            end_time = args.valid_until_time
        elif args.valid_until_duration:
            end_time = args.valid_until_duration.GetRelativeDateTime(current_time)
        if start_time and end_time:
            valid_interval = tpu_messages.Interval()
            valid_interval.startTime = times.FormatDateTime(start_time)
            valid_interval.endTime = times.FormatDateTime(end_time)
            if request.queuedResource is None:
                request.queuedResource = tpu_messages.QueuedResource()
            request.queuedResource.queueingPolicy = tpu_messages.QueueingPolicy()
            request.queuedResource.queueingPolicy.validInterval = valid_interval
        return request
    return Process