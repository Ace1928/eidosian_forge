from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingSideInputLocation(_messages.Message):
    """Identifies the location of a streaming side input.

  Fields:
    stateFamily: Identifies the state family where this side input is stored.
    tag: Identifies the particular side input within the streaming Dataflow
      job.
  """
    stateFamily = _messages.StringField(1)
    tag = _messages.StringField(2)