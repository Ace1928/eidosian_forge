from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class FormatConversionsValue(_messages.Message):
    """Optional. File format conversion map to be applied to all input files.
    Map's key is the original mime_type. Map's value is the target mime_type
    of translated documents. Supported file format conversion includes: -
    `application/pdf` to `application/vnd.openxmlformats-
    officedocument.wordprocessingml.document` If nothing specified, output
    files will be in the same format as the original file.

    Messages:
      AdditionalProperty: An additional property for a FormatConversionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        FormatConversionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a FormatConversionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)