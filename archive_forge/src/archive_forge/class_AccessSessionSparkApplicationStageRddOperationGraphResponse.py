from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessSessionSparkApplicationStageRddOperationGraphResponse(_messages.Message):
    """RDD operation graph for a Spark Application Stage limited to maximum
  10000 clusters.

  Fields:
    rddOperationGraph: RDD operation graph for a Spark Application Stage.
  """
    rddOperationGraph = _messages.MessageField('RddOperationGraph', 1)