from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetAttribute(_messages.Message):
    """Contains criteria for selecting Targets.

  Messages:
    LabelsValue: Target labels.

  Fields:
    id: ID of the `Target`. The value of this field could be one of the
      following: * The last segment of a target name. It only needs the ID to
      determine which target is being referred to * "*", all targets in a
      location.
    labels: Target labels.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Target labels.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    id = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)