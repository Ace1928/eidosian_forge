from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaxPodsConstraint(_messages.Message):
    """Constraints applied to pods.

  Fields:
    maxPodsPerNode: Constraint enforced on the max num of pods per node.
  """
    maxPodsPerNode = _messages.IntegerField(1)