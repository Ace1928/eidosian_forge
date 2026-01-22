from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpAppconnectorsV1ResourceInfo(_messages.Message):
    """ResourceInfo represents the information/status of an app connector
  resource. Such as: - remote_agent - container - runtime - appgateway -
  appconnector - appconnection - tunnel - logagent

  Enums:
    StatusValueValuesEnum: Overall health status. Overall status is derived
      based on the status of each sub level resources.

  Messages:
    ResourceValue: Specific details for the resource. This is for internal use
      only.

  Fields:
    id: Required. Unique Id for the resource.
    resource: Specific details for the resource. This is for internal use
      only.
    status: Overall health status. Overall status is derived based on the
      status of each sub level resources.
    sub: List of Info for the sub level resources.
    time: The timestamp to collect the info. It is suggested to be set by the
      topmost level resource only.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """Overall health status. Overall status is derived based on the status
    of each sub level resources.

    Values:
      HEALTH_STATUS_UNSPECIFIED: Health status is unknown: not initialized or
        failed to retrieve.
      HEALTHY: The resource is healthy.
      UNHEALTHY: The resource is unhealthy.
      UNRESPONSIVE: The resource is unresponsive.
      DEGRADED: Some sub-resources are UNHEALTHY.
    """
        HEALTH_STATUS_UNSPECIFIED = 0
        HEALTHY = 1
        UNHEALTHY = 2
        UNRESPONSIVE = 3
        DEGRADED = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourceValue(_messages.Message):
        """Specific details for the resource. This is for internal use only.

    Messages:
      AdditionalProperty: An additional property for a ResourceValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    id = _messages.StringField(1)
    resource = _messages.MessageField('ResourceValue', 2)
    status = _messages.EnumField('StatusValueValuesEnum', 3)
    sub = _messages.MessageField('GoogleCloudBeyondcorpAppconnectorsV1ResourceInfo', 4, repeated=True)
    time = _messages.StringField(5)