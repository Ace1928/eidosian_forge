from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListActiveBreakpointsResponse(_messages.Message):
    """Response for listing active breakpoints.

  Fields:
    breakpoints: List of all active breakpoints. The fields `id` and
      `location` are guaranteed to be set on each breakpoint.
    nextWaitToken: A token that can be used in the next method call to block
      until the list of breakpoints changes.
    waitExpired: If set to `true`, indicates that there is no change to the
      list of active breakpoints and the server-selected timeout has expired.
      The `breakpoints` field would be empty and should be ignored.
  """
    breakpoints = _messages.MessageField('Breakpoint', 1, repeated=True)
    nextWaitToken = _messages.StringField(2)
    waitExpired = _messages.BooleanField(3)