from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExitInfo(_messages.Message):
    """Information about how a command ended.

  Fields:
    error: Error message. Empty if there was no error.
    exitCode: The exit code from the command execution.
  """
    error = _messages.StringField(1)
    exitCode = _messages.IntegerField(2, variant=_messages.Variant.INT32)