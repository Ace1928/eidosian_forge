from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DetectLanguageRequest(_messages.Message):
    """The request message for language detection.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata for the
      request. Label keys and values can be no longer than 63 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. Label values are optional. Label keys must start with a letter.
      See https://cloud.google.com/translate/docs/labels for more information.

  Fields:
    content: The content of the input stored as a string.
    labels: Optional. The labels with user-defined metadata for the request.
      Label keys and values can be no longer than 63 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. Label
      values are optional. Label keys must start with a letter. See
      https://cloud.google.com/translate/docs/labels for more information.
    mimeType: Optional. The format of the source text, for example,
      "text/html", "text/plain". If left blank, the MIME type defaults to
      "text/html".
    model: Optional. The language detection model to be used. Format:
      `projects/{project-number-or-id}/locations/{location-
      id}/models/language-detection/{model-id}` Only one language detection
      model is currently supported: `projects/{project-number-or-
      id}/locations/{location-id}/models/language-detection/default`. If not
      specified, the default model is used.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata for the request. Label
    keys and values can be no longer than 63 characters (Unicode codepoints),
    can only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. Label values are optional.
    Label keys must start with a letter. See
    https://cloud.google.com/translate/docs/labels for more information.

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
    content = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)
    mimeType = _messages.StringField(3)
    model = _messages.StringField(4)