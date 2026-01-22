from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class TranscriptionMetadataValue(_messages.Message):
    """Map from provided filename to the transcription metadata for that
    file.

    Messages:
      AdditionalProperty: An additional property for a
        TranscriptionMetadataValue object.

    Fields:
      additionalProperties: Additional properties of type
        TranscriptionMetadataValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a TranscriptionMetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A BatchRecognizeTranscriptionMetadata attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('BatchRecognizeTranscriptionMetadata', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)