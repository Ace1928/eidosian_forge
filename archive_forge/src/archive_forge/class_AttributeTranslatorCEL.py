from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttributeTranslatorCEL(_messages.Message):
    """Specifies a list of output attribute names and the corresponding input
  attribute to use for that output attribute. Each defined output attribute is
  populated with the value of the specified input attribute.

  Messages:
    AttributesValue: Each entry specifies the desired output attribute and a
      CEL field selector expression for the corresponding input to read. This
      field supports a subset of the CEL functionality to select fields from
      the input (no boolean expressions, functions or arithmetics). Output
      attributes must match `(google.sub|a-z_*)`. The output attribute
      google.sub is interpreted to be the "identity" of the requesting user.
      For example, to copy the inbound attribute "sub" into the output
      `google.sub` add an entry `google.sub` -> `inclaim.sub` or `google.sub`
      -> `inclaim[\\"sub\\"]`. See https://github.com/google/cel-spec for more
      details. If the input does not exist the output attribute will be null.

  Fields:
    attributes: Each entry specifies the desired output attribute and a CEL
      field selector expression for the corresponding input to read. This
      field supports a subset of the CEL functionality to select fields from
      the input (no boolean expressions, functions or arithmetics). Output
      attributes must match `(google.sub|a-z_*)`. The output attribute
      google.sub is interpreted to be the "identity" of the requesting user.
      For example, to copy the inbound attribute "sub" into the output
      `google.sub` add an entry `google.sub` -> `inclaim.sub` or `google.sub`
      -> `inclaim[\\"sub\\"]`. See https://github.com/google/cel-spec for more
      details. If the input does not exist the output attribute will be null.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributesValue(_messages.Message):
        """Each entry specifies the desired output attribute and a CEL field
    selector expression for the corresponding input to read. This field
    supports a subset of the CEL functionality to select fields from the input
    (no boolean expressions, functions or arithmetics). Output attributes must
    match `(google.sub|a-z_*)`. The output attribute google.sub is interpreted
    to be the "identity" of the requesting user. For example, to copy the
    inbound attribute "sub" into the output `google.sub` add an entry
    `google.sub` -> `inclaim.sub` or `google.sub` -> `inclaim[\\"sub\\"]`. See
    https://github.com/google/cel-spec for more details. If the input does not
    exist the output attribute will be null.

    Messages:
      AdditionalProperty: An additional property for a AttributesValue object.

    Fields:
      additionalProperties: Additional properties of type AttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attributes = _messages.MessageField('AttributesValue', 1)