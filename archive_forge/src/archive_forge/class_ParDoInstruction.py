from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParDoInstruction(_messages.Message):
    """An instruction that does a ParDo operation. Takes one main input and
  zero or more side inputs, and produces zero or more outputs. Runs user code.

  Messages:
    UserFnValue: The user function to invoke.

  Fields:
    input: The input.
    multiOutputInfos: Information about each of the outputs, if user_fn is a
      MultiDoFn.
    numOutputs: The number of outputs.
    sideInputs: Zero or more side inputs.
    userFn: The user function to invoke.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class UserFnValue(_messages.Message):
        """The user function to invoke.

    Messages:
      AdditionalProperty: An additional property for a UserFnValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a UserFnValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    input = _messages.MessageField('InstructionInput', 1)
    multiOutputInfos = _messages.MessageField('MultiOutputInfo', 2, repeated=True)
    numOutputs = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    sideInputs = _messages.MessageField('SideInputInfo', 4, repeated=True)
    userFn = _messages.MessageField('UserFnValue', 5)