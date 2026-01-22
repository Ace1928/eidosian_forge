from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeqMapTaskOutputInfo(_messages.Message):
    """Information about an output of a SeqMapTask.

  Fields:
    sink: The sink to write the output value to.
    tag: The id of the TupleTag the user code will tag the output value by.
  """
    sink = _messages.MessageField('Sink', 1)
    tag = _messages.StringField(2)