from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PurposeDataValue(_messages.Message):
    """Optional. Purpose data corresponds to the policy system that the tag
    is intended for. See documentation for `Purpose` for formatting of this
    field. Purpose data cannot be changed once set.

    Messages:
      AdditionalProperty: An additional property for a PurposeDataValue
        object.

    Fields:
      additionalProperties: Additional properties of type PurposeDataValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PurposeDataValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)