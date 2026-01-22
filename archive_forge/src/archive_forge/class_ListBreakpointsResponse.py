from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListBreakpointsResponse(_messages.Message):
    """Response for listing breakpoints.

  Fields:
    breakpoints: List of breakpoints matching the request. The fields `id` and
      `location` are guaranteed to be set on each breakpoint. The fields:
      `stack_frames`, `evaluated_expressions` and `variable_table` are cleared
      on each breakpoint regardless of its status.
    nextWaitToken: A wait token that can be used in the next call to `list`
      (REST) or `ListBreakpoints` (RPC) to block until the list of breakpoints
      has changes.
  """
    breakpoints = _messages.MessageField('Breakpoint', 1, repeated=True)
    nextWaitToken = _messages.StringField(2)