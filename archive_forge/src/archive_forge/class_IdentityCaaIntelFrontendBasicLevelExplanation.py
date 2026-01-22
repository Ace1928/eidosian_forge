from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityCaaIntelFrontendBasicLevelExplanation(_messages.Message):
    """The Explanation of a Basic Level, which contains the explanation in
  Struct NextTAG: 2

  Messages:
    ExplanationValue: A ExplanationValue object.

  Fields:
    explanation: A ExplanationValue attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExplanationValue(_messages.Message):
        """A ExplanationValue object.

    Messages:
      AdditionalProperty: An additional property for a ExplanationValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExplanationValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    explanation = _messages.MessageField('ExplanationValue', 1)