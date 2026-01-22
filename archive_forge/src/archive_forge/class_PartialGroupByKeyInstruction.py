from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartialGroupByKeyInstruction(_messages.Message):
    """An instruction that does a partial group-by-key. One input and one
  output.

  Messages:
    InputElementCodecValue: The codec to use for interpreting an element in
      the input PTable.
    ValueCombiningFnValue: The value combining function to invoke.

  Fields:
    input: Describes the input to the partial group-by-key instruction.
    inputElementCodec: The codec to use for interpreting an element in the
      input PTable.
    originalCombineValuesInputStoreName: If this instruction includes a
      combining function this is the name of the intermediate store between
      the GBK and the CombineValues.
    originalCombineValuesStepName: If this instruction includes a combining
      function, this is the name of the CombineValues instruction lifted into
      this instruction.
    sideInputs: Zero or more side inputs.
    valueCombiningFn: The value combining function to invoke.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InputElementCodecValue(_messages.Message):
        """The codec to use for interpreting an element in the input PTable.

    Messages:
      AdditionalProperty: An additional property for a InputElementCodecValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InputElementCodecValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ValueCombiningFnValue(_messages.Message):
        """The value combining function to invoke.

    Messages:
      AdditionalProperty: An additional property for a ValueCombiningFnValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ValueCombiningFnValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    input = _messages.MessageField('InstructionInput', 1)
    inputElementCodec = _messages.MessageField('InputElementCodecValue', 2)
    originalCombineValuesInputStoreName = _messages.StringField(3)
    originalCombineValuesStepName = _messages.StringField(4)
    sideInputs = _messages.MessageField('SideInputInfo', 5, repeated=True)
    valueCombiningFn = _messages.MessageField('ValueCombiningFnValue', 6)