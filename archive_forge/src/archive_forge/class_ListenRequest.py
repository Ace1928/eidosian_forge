from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListenRequest(_messages.Message):
    """A request for Firestore.Listen

  Messages:
    LabelsValue: Labels associated with this target change.

  Fields:
    addTarget: A target to add to this stream.
    labels: Labels associated with this target change.
    removeTarget: The ID of a target to remove from this stream.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels associated with this target change.

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
    addTarget = _messages.MessageField('Target', 1)
    labels = _messages.MessageField('LabelsValue', 2)
    removeTarget = _messages.IntegerField(3, variant=_messages.Variant.INT32)