from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShellTask(_messages.Message):
    """A task which consists of a shell command for the worker to execute.

  Fields:
    command: The shell command to run.
    exitCode: Exit code for the task.
  """
    command = _messages.StringField(1)
    exitCode = _messages.IntegerField(2, variant=_messages.Variant.INT32)