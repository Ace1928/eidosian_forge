from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class EnvVarsValue(_messages.Message):
    """Key-value pairs to set as environment variables. Note that integration
    bindings will add/update the list of final env vars that are deployed to a
    service.

    Messages:
      AdditionalProperty: An additional property for a EnvVarsValue object.

    Fields:
      additionalProperties: Additional properties of type EnvVarsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a EnvVarsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)