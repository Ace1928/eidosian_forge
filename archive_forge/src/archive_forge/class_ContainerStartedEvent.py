from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContainerStartedEvent(_messages.Message):
    """An event generated when a container starts.

  Messages:
    PortMappingsValue: The container-to-host port mappings installed for this
      container. This set will contain any ports exposed using the
      `PUBLISH_EXPOSED_PORTS` flag as well as any specified in the `Action`
      definition.

  Fields:
    actionId: The numeric ID of the action that started this container.
    ipAddress: The public IP address that can be used to connect to the
      container. This field is only populated when at least one port mapping
      is present. If the instance was created with a private address, this
      field will be empty even if port mappings exist.
    portMappings: The container-to-host port mappings installed for this
      container. This set will contain any ports exposed using the
      `PUBLISH_EXPOSED_PORTS` flag as well as any specified in the `Action`
      definition.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PortMappingsValue(_messages.Message):
        """The container-to-host port mappings installed for this container. This
    set will contain any ports exposed using the `PUBLISH_EXPOSED_PORTS` flag
    as well as any specified in the `Action` definition.

    Messages:
      AdditionalProperty: An additional property for a PortMappingsValue
        object.

    Fields:
      additionalProperties: Additional properties of type PortMappingsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PortMappingsValue object.

      Fields:
        key: Name of the additional property.
        value: A integer attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2, variant=_messages.Variant.INT32)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    actionId = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    ipAddress = _messages.StringField(2)
    portMappings = _messages.MessageField('PortMappingsValue', 3)