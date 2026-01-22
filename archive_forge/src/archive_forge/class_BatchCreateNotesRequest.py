from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchCreateNotesRequest(_messages.Message):
    """Request to create notes in batch.

  Messages:
    NotesValue: Required. The notes to create. Max allowed length is 1000.

  Fields:
    notes: Required. The notes to create. Max allowed length is 1000.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NotesValue(_messages.Message):
        """Required. The notes to create. Max allowed length is 1000.

    Messages:
      AdditionalProperty: An additional property for a NotesValue object.

    Fields:
      additionalProperties: Additional properties of type NotesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NotesValue object.

      Fields:
        key: Name of the additional property.
        value: A Note attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('Note', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    notes = _messages.MessageField('NotesValue', 1)