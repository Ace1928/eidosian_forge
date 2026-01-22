from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ClouddebuggerControllerDebuggeesBreakpointsUpdateRequest(_messages.Message):
    """A ClouddebuggerControllerDebuggeesBreakpointsUpdateRequest object.

  Fields:
    debuggeeId: Required. Identifies the debuggee being debugged.
    id: Breakpoint identifier, unique in the scope of the debuggee.
    updateActiveBreakpointRequest: A UpdateActiveBreakpointRequest resource to
      be passed as the request body.
  """
    debuggeeId = _messages.StringField(1, required=True)
    id = _messages.StringField(2, required=True)
    updateActiveBreakpointRequest = _messages.MessageField('UpdateActiveBreakpointRequest', 3)