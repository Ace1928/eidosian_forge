from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaRuleSetting(_messages.Message):
    """Message to capture settings for a BrowserDlpRule

  Messages:
    ValueValue: Required. The value of the Setting.

  Fields:
    type: Required. Immutable. The type of the Setting. .
    value: Required. The value of the Setting.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ValueValue(_messages.Message):
        """Required. The value of the Setting.

    Messages:
      AdditionalProperty: An additional property for a ValueValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ValueValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    type = _messages.StringField(1)
    value = _messages.MessageField('ValueValue', 2)