from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GetBreakpointResponse(_messages.Message):
    """Response for getting breakpoint information.

  Fields:
    breakpoint: Complete breakpoint state. The fields `id` and `location` are
      guaranteed to be set.
  """
    breakpoint = _messages.MessageField('Breakpoint', 1)