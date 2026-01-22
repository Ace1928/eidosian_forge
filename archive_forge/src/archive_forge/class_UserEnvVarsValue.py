from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class UserEnvVarsValue(_messages.Message):
    """Optional. User-defined environment variables associated with this
    workflow revision. This map has a maximum length of 20. Each string can
    take up to 4KiB. Keys cannot be empty strings and cannot start with
    "GOOGLE" or "WORKFLOWS".

    Messages:
      AdditionalProperty: An additional property for a UserEnvVarsValue
        object.

    Fields:
      additionalProperties: Additional properties of type UserEnvVarsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a UserEnvVarsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)