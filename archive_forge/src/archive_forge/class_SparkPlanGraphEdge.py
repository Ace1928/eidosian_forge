from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkPlanGraphEdge(_messages.Message):
    """Represents a directed edge in the spark plan tree from child to parent.

  Fields:
    fromId: A string attribute.
    toId: A string attribute.
  """
    fromId = _messages.IntegerField(1)
    toId = _messages.IntegerField(2)