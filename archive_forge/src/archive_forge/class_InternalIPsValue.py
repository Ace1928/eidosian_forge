from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class InternalIPsValue(_messages.Message):
    """Internal network IPs assigned to the instances that will be preserved
    on instance delete, update, etc. This map is keyed with the network
    interface name.

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
        value: A StatefulPolicyPreservedStateNetworkIp attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('StatefulPolicyPreservedStateNetworkIp', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)