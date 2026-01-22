from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreservedState(_messages.Message):
    """Preserved state for a given instance.

  Messages:
    DisksValue: Preserved disks defined for this instance. This map is keyed
      with the device names of the disks.
    ExternalIPsValue: Preserved external IPs defined for this instance. This
      map is keyed with the name of the network interface.
    InternalIPsValue: Preserved internal IPs defined for this instance. This
      map is keyed with the name of the network interface.
    MetadataValue: Preserved metadata defined for this instance.

  Fields:
    disks: Preserved disks defined for this instance. This map is keyed with
      the device names of the disks.
    externalIPs: Preserved external IPs defined for this instance. This map is
      keyed with the name of the network interface.
    internalIPs: Preserved internal IPs defined for this instance. This map is
      keyed with the name of the network interface.
    metadata: Preserved metadata defined for this instance.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DisksValue(_messages.Message):
        """Preserved disks defined for this instance. This map is keyed with the
    device names of the disks.

    Messages:
      AdditionalProperty: An additional property for a DisksValue object.

    Fields:
      additionalProperties: Additional properties of type DisksValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DisksValue object.

      Fields:
        key: Name of the additional property.
        value: A PreservedStatePreservedDisk attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PreservedStatePreservedDisk', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExternalIPsValue(_messages.Message):
        """Preserved external IPs defined for this instance. This map is keyed
    with the name of the network interface.

    Messages:
      AdditionalProperty: An additional property for a ExternalIPsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ExternalIPsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExternalIPsValue object.

      Fields:
        key: Name of the additional property.
        value: A PreservedStatePreservedNetworkIp attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PreservedStatePreservedNetworkIp', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InternalIPsValue(_messages.Message):
        """Preserved internal IPs defined for this instance. This map is keyed
    with the name of the network interface.

    Messages:
      AdditionalProperty: An additional property for a InternalIPsValue
        object.

    Fields:
      additionalProperties: Additional properties of type InternalIPsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InternalIPsValue object.

      Fields:
        key: Name of the additional property.
        value: A PreservedStatePreservedNetworkIp attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PreservedStatePreservedNetworkIp', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Preserved metadata defined for this instance.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Additional properties of type MetadataValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    disks = _messages.MessageField('DisksValue', 1)
    externalIPs = _messages.MessageField('ExternalIPsValue', 2)
    internalIPs = _messages.MessageField('InternalIPsValue', 3)
    metadata = _messages.MessageField('MetadataValue', 4)