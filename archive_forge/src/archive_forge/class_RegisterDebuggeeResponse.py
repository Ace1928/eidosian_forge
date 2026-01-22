from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RegisterDebuggeeResponse(_messages.Message):
    """Response for registering a debuggee.

  Fields:
    agentId: A unique ID generated for the agent. Each RegisterDebuggee
      request will generate a new agent ID.
    debuggee: Debuggee resource. The field `id` is guaranteed to be set (in
      addition to the echoed fields). If the field `is_disabled` is set to
      `true`, the agent should disable itself by removing all breakpoints and
      detaching from the application. It should however continue to poll
      `RegisterDebuggee` until reenabled.
  """
    agentId = _messages.StringField(1)
    debuggee = _messages.MessageField('Debuggee', 2)