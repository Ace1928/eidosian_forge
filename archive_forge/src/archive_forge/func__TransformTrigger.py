from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.eventarc import types as trigger_types
def _TransformTrigger(data, undefined=''):
    """Returns textual information about functions trigger.

  Args:
    data: JSON-serializable 1st and 2nd gen Functions objects.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    str containing information about functions trigger.
  """
    data_type = _InferFunctionMessageFormat(data)
    if data_type == CLOUD_FUNCTION:
        if 'httpsTrigger' in data:
            return 'HTTP Trigger'
        if 'gcsTrigger' in data:
            return 'bucket: ' + data['gcsTrigger']
        if 'pubsubTrigger' in data:
            return 'topic: ' + data['pubsubTrigger'].split('/')[-1]
        if 'eventTrigger' in data:
            return 'Event Trigger'
        return undefined
    elif data_type == FUNCTION:
        if 'eventTrigger' in data:
            event_trigger = data['eventTrigger']
            event_type = event_trigger.get('eventType')
            if trigger_types.IsAuditLogType(event_type):
                return 'Cloud Audit Log'
            elif trigger_types.IsStorageType(event_type):
                event_filters = event_trigger['eventFilters']
                bucket = next((f.get('value') for f in event_filters if f.get('attribute') == 'bucket'), None)
                if bucket:
                    return 'bucket: ' + bucket
            if 'pubsubTopic' in event_trigger:
                return 'topic: ' + event_trigger['pubsubTopic'].split('/')[-1]
            return 'Event Trigger'
        return 'HTTP Trigger'
    return undefined