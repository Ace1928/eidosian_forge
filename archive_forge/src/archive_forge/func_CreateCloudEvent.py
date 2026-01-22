from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import extra_types
from googlecloudsdk.core.util import times
def CreateCloudEvent(event_id, event_type, event_source, event_data, event_attributes):
    """Transform args to a valid cloud event.

  Args:
    event_id: The id of a published event.
    event_type: The event type of a published event.
    event_source: The event source of a published event.
    event_data: The event data of a published event.
    event_attributes: The event attributes of a published event. It can be
      repeated to add more attributes.

  Returns:
    valid CloudEvent.

  """
    cloud_event = {'@type': 'type.googleapis.com/io.cloudevents.v1.CloudEvent', 'id': event_id, 'source': event_source, 'specVersion': '1.0', 'type': event_type, 'attributes': {'time': {'ceTimestamp': times.FormatDateTime(times.Now())}, 'datacontenttype': {'ceString': 'application/json'}}, 'textData': event_data}
    if event_attributes is not None:
        for key, value in event_attributes.items():
            cloud_event['attributes'][key] = {'ceString': value}
    return cloud_event