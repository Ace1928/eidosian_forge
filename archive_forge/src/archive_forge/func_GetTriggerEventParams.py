from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import exceptions
from googlecloudsdk.api_lib.functions.v1 import triggers
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetTriggerEventParams(trigger_http, trigger_bucket, trigger_topic, trigger_event, trigger_resource):
    """Check --trigger-*  arguments and deduce if possible.

  0. if --trigger-http is return None.
  1. if --trigger-bucket return bucket trigger args (_GetBucketTriggerArgs)
  2. if --trigger-topic return pub-sub trigger args (_GetTopicTriggerArgs)
  3. if --trigger-event, deduce provider and resource from registry and return

  Args:
    trigger_http: The trigger http
    trigger_bucket: The trigger bucket
    trigger_topic: The trigger topic
    trigger_event: The trigger event
    trigger_resource: The trigger resource

  Returns:
    None, when using HTTPS trigger. Otherwise a dictionary containing
    trigger_provider, trigger_event, and trigger_resource.
  """
    if trigger_http:
        return None
    if trigger_bucket:
        return _GetBucketTriggerEventParams(trigger_bucket)
    if trigger_topic:
        return _GetTopicTriggerEventParams(trigger_topic)
    if trigger_event:
        return _GetEventTriggerEventParams(trigger_event, trigger_resource)
    elif trigger_resource:
        log.warning('Ignoring the flag --trigger-resource. The flag --trigger-resource is provided but --trigger-event is not. If you intend to change trigger-resource you need to provide trigger-event as well.')