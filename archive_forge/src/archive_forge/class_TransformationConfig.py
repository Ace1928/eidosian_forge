from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransformationConfig(_messages.Message):
    """A TransformationConfig configures the associated AssetType to invoke
  transformers on its Assets.

  Messages:
    InputsValue: A InputsValue object.
    OutputsValue: Required. Key-value pairs representing output parameters
      from the transformers. The key maps to the transformer output parameter
      name. The value will be the path to the metadata in the asset to which
      this output should be assigned.

  Fields:
    inputs: A InputsValue attribute.
    outputs: Required. Key-value pairs representing output parameters from the
      transformers. The key maps to the transformer output parameter name. The
      value will be the path to the metadata in the asset to which this output
      should be assigned.
    transformer: Required. Reference to a transformer to execute, in the
      following form:
      `projects/{project}/locations/{location}/transformers/{name}`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InputsValue(_messages.Message):
        """A InputsValue object.

    Messages:
      AdditionalProperty: An additional property for a InputsValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InputsValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class OutputsValue(_messages.Message):
        """Required. Key-value pairs representing output parameters from the
    transformers. The key maps to the transformer output parameter name. The
    value will be the path to the metadata in the asset to which this output
    should be assigned.

    Messages:
      AdditionalProperty: An additional property for a OutputsValue object.

    Fields:
      additionalProperties: Additional properties of type OutputsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a OutputsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    inputs = _messages.MessageField('InputsValue', 1)
    outputs = _messages.MessageField('OutputsValue', 2)
    transformer = _messages.StringField(3)