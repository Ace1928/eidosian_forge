from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ArgumentsValue(_messages.Message):
    """Collection of all external inputs that influenced the build on top of
    recipe.definedInMaterial and recipe.entryPoint. For example, if the recipe
    type were "make", then this might be the flags passed to make aside from
    the target, which is captured in recipe.entryPoint. Depending on the
    recipe Type, the structure may be different.

    Messages:
      AdditionalProperty: An additional property for a ArgumentsValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ArgumentsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)