from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class BuildEnvVariablesValue(_messages.Message):
    """Environment variables available to the build environment.Only returned
    in GET requests if view=FULL is set.

    Messages:
      AdditionalProperty: An additional property for a BuildEnvVariablesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        BuildEnvVariablesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a BuildEnvVariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)