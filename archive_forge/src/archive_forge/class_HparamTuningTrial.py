from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HparamTuningTrial(_messages.Message):
    """Training info of a trial in [hyperparameter tuning](/bigquery-
  ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview) models.

  Enums:
    StatusValueValuesEnum: The status of the trial.

  Fields:
    endTimeMs: Ending time of the trial.
    errorMessage: Error message for FAILED and INFEASIBLE trial.
    evalLoss: Loss computed on the eval data at the end of trial.
    evaluationMetrics: Evaluation metrics of this trial calculated on the test
      data. Empty in Job API.
    hparamTuningEvaluationMetrics: Hyperparameter tuning evaluation metrics of
      this trial calculated on the eval data. Unlike evaluation_metrics, only
      the fields corresponding to the hparam_tuning_objectives are set.
    hparams: The hyperprameters selected for this trial.
    startTimeMs: Starting time of the trial.
    status: The status of the trial.
    trainingLoss: Loss computed on the training data at the end of trial.
    trialId: 1-based index of the trial.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """The status of the trial.

    Values:
      TRIAL_STATUS_UNSPECIFIED: Default value.
      NOT_STARTED: Scheduled but not started.
      RUNNING: Running state.
      SUCCEEDED: The trial succeeded.
      FAILED: The trial failed.
      INFEASIBLE: The trial is infeasible due to the invalid params.
      STOPPED_EARLY: Trial stopped early because it's not promising.
    """
        TRIAL_STATUS_UNSPECIFIED = 0
        NOT_STARTED = 1
        RUNNING = 2
        SUCCEEDED = 3
        FAILED = 4
        INFEASIBLE = 5
        STOPPED_EARLY = 6
    endTimeMs = _messages.IntegerField(1)
    errorMessage = _messages.StringField(2)
    evalLoss = _messages.FloatField(3)
    evaluationMetrics = _messages.MessageField('EvaluationMetrics', 4)
    hparamTuningEvaluationMetrics = _messages.MessageField('EvaluationMetrics', 5)
    hparams = _messages.MessageField('TrainingOptions', 6)
    startTimeMs = _messages.IntegerField(7)
    status = _messages.EnumField('StatusValueValuesEnum', 8)
    trainingLoss = _messages.FloatField(9)
    trialId = _messages.IntegerField(10)