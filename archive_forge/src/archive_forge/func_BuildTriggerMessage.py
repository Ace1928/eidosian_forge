from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.eventarc.base import EventarcClientBase
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.eventarc import types
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times
def BuildTriggerMessage(self, trigger_ref, event_filters, event_filters_path_pattern, event_data_content_type, service_account, destination_message, transport_topic_ref, channel_ref):
    """Builds a Trigger message with the given data.

    Args:
      trigger_ref: Resource, the Trigger to create.
      event_filters: dict or None, the Trigger's event filters.
      event_filters_path_pattern: dict or None, the Trigger's event filters in
        path pattern format. Ignored in Beta.
      event_data_content_type: str or None, the data content type of the
        Trigger's event. Ignored in Beta.
      service_account: str or None, the Trigger's service account.
      destination_message: Destination message or None, the Trigger's
        destination.
      transport_topic_ref: Resource or None, the user-provided transport topic.
      channel_ref: Resource or None, the channel for 3p events. Ignored in Beta.

    Returns:
      A Trigger message with a destination service.
    """
    criteria_messages = [] if event_filters is None else [self._messages.MatchingCriteria(attribute=key, value=value) for key, value in event_filters.items()]
    transport = None
    if transport_topic_ref:
        transport_topic_name = transport_topic_ref.RelativeName()
        pubsub = self._messages.Pubsub(topic=transport_topic_name)
        transport = self._messages.Transport(pubsub=pubsub)
    return self._messages.Trigger(name=trigger_ref.RelativeName(), matchingCriteria=criteria_messages, serviceAccount=service_account, destination=destination_message, transport=transport)