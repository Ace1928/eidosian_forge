from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessSessionSparkApplicationSqlSparkPlanGraphResponse(_messages.Message):
    """SparkPlanGraph for a Spark Application execution limited to maximum
  10000 clusters.

  Fields:
    sparkPlanGraph: SparkPlanGraph for a Spark Application execution.
  """
    sparkPlanGraph = _messages.MessageField('SparkPlanGraph', 1)