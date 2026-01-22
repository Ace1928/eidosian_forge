from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StudySpecMetricSpec(_messages.Message):
    """Represents a metric to optimize.

  Enums:
    GoalValueValuesEnum: Required. The optimization goal of the metric.

  Fields:
    goal: Required. The optimization goal of the metric.
    metricId: Required. The ID of the metric. Must not contain whitespaces and
      must be unique amongst all MetricSpecs.
    safetyConfig: Used for safe search. In the case, the metric will be a
      safety metric. You must provide a separate metric for objective metric.
  """

    class GoalValueValuesEnum(_messages.Enum):
        """Required. The optimization goal of the metric.

    Values:
      GOAL_TYPE_UNSPECIFIED: Goal Type will default to maximize.
      MAXIMIZE: Maximize the goal metric.
      MINIMIZE: Minimize the goal metric.
    """
        GOAL_TYPE_UNSPECIFIED = 0
        MAXIMIZE = 1
        MINIMIZE = 2
    goal = _messages.EnumField('GoalValueValuesEnum', 1)
    metricId = _messages.StringField(2)
    safetyConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecMetricSpecSafetyMetricConfig', 3)