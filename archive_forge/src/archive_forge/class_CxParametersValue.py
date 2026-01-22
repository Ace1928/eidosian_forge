from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class CxParametersValue(_messages.Message):
    """Additional parameters to be put into Dialogflow CX session parameters.
    To remove a parameter from the session, clients should explicitly set the
    parameter value to null. Note: this field should only be used if you are
    connecting to a Dialogflow CX agent.

    Messages:
      AdditionalProperty: An additional property for a CxParametersValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a CxParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)