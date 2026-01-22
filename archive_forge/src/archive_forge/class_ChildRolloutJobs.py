from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChildRolloutJobs(_messages.Message):
    """ChildRollouts job composition

  Fields:
    advanceRolloutJobs: Output only. List of AdvanceChildRolloutJobs
    createRolloutJobs: Output only. List of CreateChildRolloutJobs
  """
    advanceRolloutJobs = _messages.MessageField('Job', 1, repeated=True)
    createRolloutJobs = _messages.MessageField('Job', 2, repeated=True)