from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpAppconnectorsV1AppConnectorInstanceConfig(_messages.Message):
    """AppConnectorInstanceConfig defines the instance config of a
  AppConnector.

  Messages:
    InstanceConfigValue: The SLM instance agent configuration.

  Fields:
    imageConfig: ImageConfig defines the GCR images to run for the remote
      agent's control plane.
    instanceConfig: The SLM instance agent configuration.
    notificationConfig: NotificationConfig defines the notification mechanism
      that the remote instance should subscribe to in order to receive
      notification.
    sequenceNumber: Required. A monotonically increasing number generated and
      maintained by the API provider. Every time a config changes in the
      backend, the sequenceNumber should be bumped up to reflect the change.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InstanceConfigValue(_messages.Message):
        """The SLM instance agent configuration.

    Messages:
      AdditionalProperty: An additional property for a InstanceConfigValue
        object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InstanceConfigValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    imageConfig = _messages.MessageField('GoogleCloudBeyondcorpAppconnectorsV1ImageConfig', 1)
    instanceConfig = _messages.MessageField('InstanceConfigValue', 2)
    notificationConfig = _messages.MessageField('GoogleCloudBeyondcorpAppconnectorsV1NotificationConfig', 3)
    sequenceNumber = _messages.IntegerField(4)