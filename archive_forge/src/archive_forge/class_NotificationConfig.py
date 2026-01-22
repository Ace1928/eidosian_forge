from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotificationConfig(_messages.Message):
    """Specification to configure notifications published to Pub/Sub.
  Notifications are published to the customer-provided topic using the
  following `PubsubMessage.attributes`: * `"eventType"`: one of the EventType
  values * `"payloadFormat"`: one of the PayloadFormat values * `"projectId"`:
  the project_id of the `TransferOperation` * `"transferJobName"`: the
  transfer_job_name of the `TransferOperation` * `"transferOperationName"`:
  the name of the `TransferOperation` The `PubsubMessage.data` contains a
  TransferOperation resource formatted according to the specified
  `PayloadFormat`.

  Enums:
    EventTypesValueListEntryValuesEnum:
    PayloadFormatValueValuesEnum: Required. The desired format of the
      notification message payloads.

  Fields:
    eventTypes: Event types for which a notification is desired. If empty,
      send notifications for all event types.
    payloadFormat: Required. The desired format of the notification message
      payloads.
    pubsubTopic: Required. The `Topic.name` of the Pub/Sub topic to which to
      publish notifications. Must be of the format:
      `projects/{project}/topics/{topic}`. Not matching this format results in
      an INVALID_ARGUMENT error.
  """

    class EventTypesValueListEntryValuesEnum(_messages.Enum):
        """EventTypesValueListEntryValuesEnum enum type.

    Values:
      EVENT_TYPE_UNSPECIFIED: Illegal value, to avoid allowing a default.
      TRANSFER_OPERATION_SUCCESS: `TransferOperation` completed with status
        SUCCESS.
      TRANSFER_OPERATION_FAILED: `TransferOperation` completed with status
        FAILED.
      TRANSFER_OPERATION_ABORTED: `TransferOperation` completed with status
        ABORTED.
    """
        EVENT_TYPE_UNSPECIFIED = 0
        TRANSFER_OPERATION_SUCCESS = 1
        TRANSFER_OPERATION_FAILED = 2
        TRANSFER_OPERATION_ABORTED = 3

    class PayloadFormatValueValuesEnum(_messages.Enum):
        """Required. The desired format of the notification message payloads.

    Values:
      PAYLOAD_FORMAT_UNSPECIFIED: Illegal value, to avoid allowing a default.
      NONE: No payload is included with the notification.
      JSON: `TransferOperation` is [formatted as a JSON
        response](https://developers.google.com/protocol-
        buffers/docs/proto3#json), in application/json.
    """
        PAYLOAD_FORMAT_UNSPECIFIED = 0
        NONE = 1
        JSON = 2
    eventTypes = _messages.EnumField('EventTypesValueListEntryValuesEnum', 1, repeated=True)
    payloadFormat = _messages.EnumField('PayloadFormatValueValuesEnum', 2)
    pubsubTopic = _messages.StringField(3)