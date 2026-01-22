from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DisksValue(_messages.Message):
    """Disks created on the instances that will be preserved on instance
    delete, update, etc. This map is keyed with the device names of the disks.

    Messages:
      AdditionalProperty: An additional property for a DisksValue object.

    Fields:
      additionalProperties: Additional properties of type DisksValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DisksValue object.

      Fields:
        key: Name of the additional property.
        value: A StatefulPolicyPreservedStateDiskDevice attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('StatefulPolicyPreservedStateDiskDevice', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)