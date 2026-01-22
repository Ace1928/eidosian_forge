from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DoubleCandidates(_messages.Message):
    """Discrete candidates of a double hyperparameter.

  Fields:
    candidates: Candidates for the double parameter in increasing order.
  """
    candidates = _messages.FloatField(1, repeated=True)