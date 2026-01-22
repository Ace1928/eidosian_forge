from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StudySpecConvexStopConfig(_messages.Message):
    """Configuration for ConvexStopPolicy.

  Fields:
    autoregressiveOrder: The number of Trial measurements used in
      autoregressive model for value prediction. A trial won't be considered
      early stopping if has fewer measurement points.
    learningRateParameterName: The hyper-parameter name used in the tuning job
      that stands for learning rate. Leave it blank if learning rate is not in
      a parameter in tuning. The learning_rate is used to estimate the
      objective value of the ongoing trial.
    maxNumSteps: Steps used in predicting the final objective for early
      stopped trials. In general, it's set to be the same as the defined steps
      in training / tuning. When use_steps is false, this field is set to the
      maximum elapsed seconds.
    minNumSteps: Minimum number of steps for a trial to complete. Trials which
      do not have a measurement with num_steps > min_num_steps won't be
      considered for early stopping. It's ok to set it to 0, and a trial can
      be early stopped at any stage. By default, min_num_steps is set to be
      one-tenth of the max_num_steps. When use_steps is false, this field is
      set to the minimum elapsed seconds.
    useSeconds: This bool determines whether or not the rule is applied based
      on elapsed_secs or steps. If use_seconds==false, the early stopping
      decision is made according to the predicted objective values according
      to the target steps. If use_seconds==true, elapsed_secs is used instead
      of steps. Also, in this case, the parameters max_num_steps and
      min_num_steps are overloaded to contain max_elapsed_seconds and
      min_elapsed_seconds.
  """
    autoregressiveOrder = _messages.IntegerField(1)
    learningRateParameterName = _messages.StringField(2)
    maxNumSteps = _messages.IntegerField(3)
    minNumSteps = _messages.IntegerField(4)
    useSeconds = _messages.BooleanField(5)