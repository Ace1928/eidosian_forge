from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FlattenInstruction(_messages.Message):
    """An instruction that copies its inputs (zero or more) to its (single)
  output.

  Fields:
    inputs: Describes the inputs to the flatten instruction.
  """
    inputs = _messages.MessageField('InstructionInput', 1, repeated=True)