from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkPlanGraphCluster(_messages.Message):
    """Represents a tree of spark plan.

  Fields:
    desc: A string attribute.
    metrics: A SqlPlanMetric attribute.
    name: A string attribute.
    nodes: A SparkPlanGraphNodeWrapper attribute.
    sparkPlanGraphClusterId: A string attribute.
  """
    desc = _messages.StringField(1)
    metrics = _messages.MessageField('SqlPlanMetric', 2, repeated=True)
    name = _messages.StringField(3)
    nodes = _messages.MessageField('SparkPlanGraphNodeWrapper', 4, repeated=True)
    sparkPlanGraphClusterId = _messages.IntegerField(5)