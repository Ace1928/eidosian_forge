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
def SetBestEffort(ref, args, request):
    """Creates an empty BestEffort structure if best-effort arg flag is set."""
    del ref
    if args.best_effort:
        tpu_messages = GetMessagesModule('v2alpha1')
        if request.queuedResource is None:
            request.queuedResource = tpu_messages.QueuedResource()
        if request.queuedResource.bestEffort is None:
            request.queuedResource.bestEffort = tpu_messages.BestEffort()
    return request