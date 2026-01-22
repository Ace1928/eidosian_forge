from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RegisterDebuggeeRequest(_messages.Message):
    """Request to register a debuggee.

  Fields:
    debuggee: Required. Debuggee information to register. The fields
      `project`, `uniquifier`, `description` and `agent_version` of the
      debuggee must be set.
  """
    debuggee = _messages.MessageField('Debuggee', 1)