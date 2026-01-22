from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class VariantsValue(_messages.Message):
    """Optional. variants represents the variants of the resource bundle in
    this release. key in the map represents the name of the variant.

    Messages:
      AdditionalProperty: An additional property for a VariantsValue object.

    Fields:
      additionalProperties: Additional properties of type VariantsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a VariantsValue object.

      Fields:
        key: Name of the additional property.
        value: A Variant attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Variant', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)