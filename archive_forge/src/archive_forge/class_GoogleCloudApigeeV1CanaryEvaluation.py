from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1CanaryEvaluation(_messages.Message):
    """CanaryEvaluation represents the canary analysis between two versions of
  the runtime that is serving requests.

  Enums:
    StateValueValuesEnum: Output only. The current state of the canary
      evaluation.
    VerdictValueValuesEnum: Output only. The resulting verdict of the canary
      evaluations: NONE, PASS, or FAIL.

  Fields:
    control: Required. The stable version that is serving requests.
    createTime: Output only. Create time of the canary evaluation.
    endTime: Required. End time for the evaluation's analysis.
    metricLabels: Required. Labels used to filter the metrics used for a
      canary evaluation.
    name: Output only. Name of the canary evalution.
    startTime: Required. Start time for the canary evaluation's analysis.
    state: Output only. The current state of the canary evaluation.
    treatment: Required. The newer version that is serving requests.
    verdict: Output only. The resulting verdict of the canary evaluations:
      NONE, PASS, or FAIL.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the canary evaluation.

    Values:
      STATE_UNSPECIFIED: No state has been specified.
      RUNNING: The canary evaluation is still in progress.
      SUCCEEDED: The canary evaluation has finished.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        SUCCEEDED = 2

    class VerdictValueValuesEnum(_messages.Enum):
        """Output only. The resulting verdict of the canary evaluations: NONE,
    PASS, or FAIL.

    Values:
      VERDICT_UNSPECIFIED: Verdict is not available yet.
      NONE: No verdict reached.
      FAIL: Evaluation is not good.
      PASS: Evaluation is good.
    """
        VERDICT_UNSPECIFIED = 0
        NONE = 1
        FAIL = 2
        PASS = 3
    control = _messages.StringField(1)
    createTime = _messages.StringField(2)
    endTime = _messages.StringField(3)
    metricLabels = _messages.MessageField('GoogleCloudApigeeV1CanaryEvaluationMetricLabels', 4)
    name = _messages.StringField(5)
    startTime = _messages.StringField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    treatment = _messages.StringField(8)
    verdict = _messages.EnumField('VerdictValueValuesEnum', 9)