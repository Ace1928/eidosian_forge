from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrainingTypeValueValuesEnum(_messages.Enum):
    """Output only. Training type of the job.

    Values:
      TRAINING_TYPE_UNSPECIFIED: Unspecified training type.
      SINGLE_TRAINING: Single training with fixed parameter space.
      HPARAM_TUNING: [Hyperparameter tuning training](/bigquery-
        ml/docs/reference/standard-sql/bigqueryml-syntax-hp-tuning-overview).
    """
    TRAINING_TYPE_UNSPECIFIED = 0
    SINGLE_TRAINING = 1
    HPARAM_TUNING = 2