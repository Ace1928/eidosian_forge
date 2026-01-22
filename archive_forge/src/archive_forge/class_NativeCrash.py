from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NativeCrash(_messages.Message):
    """Additional details for a native crash.

  Fields:
    stackTrace: The stack trace of the native crash. Optional.
  """
    stackTrace = _messages.MessageField('StackTrace', 1)