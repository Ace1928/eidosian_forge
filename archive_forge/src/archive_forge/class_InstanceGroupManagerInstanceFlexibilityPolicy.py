from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerInstanceFlexibilityPolicy(_messages.Message):
    """A InstanceGroupManagerInstanceFlexibilityPolicy object.

  Messages:
    InstanceSelectionListsValue: Named instance selections configuring
      properties that the group will use when creating new VMs.
    InstanceSelectionsValue: Named instance selections configuring properties
      that the group will use when creating new VMs.

  Fields:
    instanceSelectionLists: Named instance selections configuring properties
      that the group will use when creating new VMs.
    instanceSelections: Named instance selections configuring properties that
      the group will use when creating new VMs.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InstanceSelectionListsValue(_messages.Message):
        """Named instance selections configuring properties that the group will
    use when creating new VMs.

    Messages:
      AdditionalProperty: An additional property for a
        InstanceSelectionListsValue object.

    Fields:
      additionalProperties: Additional properties of type
        InstanceSelectionListsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InstanceSelectionListsValue object.

      Fields:
        key: Name of the additional property.
        value: A
          InstanceGroupManagerInstanceFlexibilityPolicyInstanceSelection
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('InstanceGroupManagerInstanceFlexibilityPolicyInstanceSelection', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InstanceSelectionsValue(_messages.Message):
        """Named instance selections configuring properties that the group will
    use when creating new VMs.

    Messages:
      AdditionalProperty: An additional property for a InstanceSelectionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        InstanceSelectionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InstanceSelectionsValue object.

      Fields:
        key: Name of the additional property.
        value: A
          InstanceGroupManagerInstanceFlexibilityPolicyInstanceSelection
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('InstanceGroupManagerInstanceFlexibilityPolicyInstanceSelection', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    instanceSelectionLists = _messages.MessageField('InstanceSelectionListsValue', 1)
    instanceSelections = _messages.MessageField('InstanceSelectionsValue', 2)