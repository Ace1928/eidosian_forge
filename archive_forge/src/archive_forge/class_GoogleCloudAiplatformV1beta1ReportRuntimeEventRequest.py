from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReportRuntimeEventRequest(_messages.Message):
    """Request message for NotebookInternalService.ReportRuntimeEvent.

  Enums:
    EventTypeValueValuesEnum: Required. The type of the event.

  Messages:
    EventDetailsValue: Optional. The details of the request for debug.

  Fields:
    eventDetails: Optional. The details of the request for debug.
    eventType: Required. The type of the event.
    internalOsServiceStateInstance: The details of the internal os service
      states.
    internalOsServiceStateInstances: Optional. The details of the internal os
      service states.
    vmToken: Required. The VM identity token (a JWT) for authenticating the
      VM. https://cloud.google.com/compute/docs/instances/verifying-instance-
      identity
  """

    class EventTypeValueValuesEnum(_messages.Enum):
        """Required. The type of the event.

    Values:
      EVENT_TYPE_UNSPECIFIED: Unspecified.
      HEARTBEAT: Used for readiness reporting.
      IDLE: Used for idle reporting.
    """
        EVENT_TYPE_UNSPECIFIED = 0
        HEARTBEAT = 1
        IDLE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EventDetailsValue(_messages.Message):
        """Optional. The details of the request for debug.

    Messages:
      AdditionalProperty: An additional property for a EventDetailsValue
        object.

    Fields:
      additionalProperties: Additional properties of type EventDetailsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EventDetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    eventDetails = _messages.MessageField('EventDetailsValue', 1)
    eventType = _messages.EnumField('EventTypeValueValuesEnum', 2)
    internalOsServiceStateInstance = _messages.MessageField('GoogleCloudAiplatformV1beta1InternalOsServiceStateInstance', 3, repeated=True)
    internalOsServiceStateInstances = _messages.MessageField('GoogleCloudAiplatformV1beta1InternalOsServiceStateInstance', 4, repeated=True)
    vmToken = _messages.StringField(5)