from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ListTrialsResponse(_messages.Message):
    """The response message for the ListTrials method.

  Fields:
    trials: The trials associated with the study.
  """
    trials = _messages.MessageField('GoogleCloudMlV1Trial', 1, repeated=True)