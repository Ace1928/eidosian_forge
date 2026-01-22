from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DoubleHparamSearchSpace(_messages.Message):
    """Search space for a double hyperparameter.

  Fields:
    candidates: Candidates of the double hyperparameter.
    range: Range of the double hyperparameter.
  """
    candidates = _messages.MessageField('DoubleCandidates', 1)
    range = _messages.MessageField('DoubleRange', 2)