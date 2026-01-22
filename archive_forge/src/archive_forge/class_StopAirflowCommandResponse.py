from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StopAirflowCommandResponse(_messages.Message):
    """Response to StopAirflowCommandRequest.

  Fields:
    isDone: Whether the execution is still running.
    output: Output message from stopping execution request.
  """
    isDone = _messages.BooleanField(1)
    output = _messages.StringField(2, repeated=True)