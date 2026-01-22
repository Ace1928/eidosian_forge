from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AuthValue(_messages.Message):
    """Authentication information.

    Messages:
      Oauth2Value: OAuth 2.0 authentication information.

    Fields:
      oauth2: OAuth 2.0 authentication information.
    """

    class Oauth2Value(_messages.Message):
        """OAuth 2.0 authentication information.

      Messages:
        ScopesValue: Available OAuth 2.0 scopes.

      Fields:
        scopes: Available OAuth 2.0 scopes.
      """

        @encoding.MapUnrecognizedFields('additionalProperties')
        class ScopesValue(_messages.Message):
            """Available OAuth 2.0 scopes.

        Messages:
          AdditionalProperty: An additional property for a ScopesValue object.

        Fields:
          additionalProperties: The scope value.
        """

            class AdditionalProperty(_messages.Message):
                """An additional property for a ScopesValue object.

          Messages:
            ValueValue: A ValueValue object.

          Fields:
            key: Name of the additional property.
            value: A ValueValue attribute.
          """

                class ValueValue(_messages.Message):
                    """A ValueValue object.

            Fields:
              description: Description of scope.
            """
                    description = _messages.StringField(1)
                key = _messages.StringField(1)
                value = _messages.MessageField('ValueValue', 2)
            additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
        scopes = _messages.MessageField('ScopesValue', 1)
    oauth2 = _messages.MessageField('Oauth2Value', 1)