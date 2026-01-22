from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import extra_types
from googlecloudsdk.core.util import times
def TransformEventsForPublishing(events_value_list_entry, cloud_event):
    """Format events value list entry into CloudEvent.

  Args:
    events_value_list_entry: A EventsValueListEntry object.
    cloud_event: A CloudEvent representation to be passed as the request body.

  Returns:
    The CloudEvents v1.0 events to publish.
  """
    proto_json = extra_types.JsonProtoDecoder(json.dumps(cloud_event))
    additional_properties = [events_value_list_entry.AdditionalProperty(key=obj.key, value=obj.value) for obj in proto_json.properties]
    return events_value_list_entry(additionalProperties=additional_properties)