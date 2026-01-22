from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolExitCode(_messages.Message):
    """Exit code from a tool execution.

  Fields:
    number: Tool execution exit code. A value of 0 means that the execution
      was successful. - In response: always set - In create/update request:
      always set
  """
    number = _messages.IntegerField(1, variant=_messages.Variant.INT32)