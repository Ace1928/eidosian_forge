from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1StudySpecStudyStoppingConfig(_messages.Message):
    """The configuration (stopping conditions) for automated stopping of a
  Study. Conditions include trial budgets, time budgets, and convergence
  detection.

  Fields:
    maxDurationNoProgress: If the objective value has not improved for this
      much time, stop the study. WARNING: Effective only for single-objective
      studies.
    maxNumTrials: If there are more than this many trials, stop the study.
    maxNumTrialsNoProgress: If the objective value has not improved for this
      many consecutive trials, stop the study. WARNING: Effective only for
      single-objective studies.
    maximumRuntimeConstraint: If the specified time or duration has passed,
      stop the study.
    minNumTrials: If there are fewer than this many COMPLETED trials, do not
      stop the study.
    minimumRuntimeConstraint: Each "stopping rule" in this proto specifies an
      "if" condition. Before Vizier would generate a new suggestion, it first
      checks each specified stopping rule, from top to bottom in this list.
      Note that the first few rules (e.g. minimum_runtime_constraint,
      min_num_trials) will prevent other stopping rules from being evaluated
      until they are met. For example, setting `min_num_trials=5` and
      `always_stop_after= 1 hour` means that the Study will ONLY stop after it
      has 5 COMPLETED trials, even if more than an hour has passed since its
      creation. It follows the first applicable rule (whose "if" condition is
      satisfied) to make a stopping decision. If none of the specified rules
      are applicable, then Vizier decides that the study should not stop. If
      Vizier decides that the study should stop, the study enters STOPPING
      state (or STOPPING_ASAP if should_stop_asap = true). IMPORTANT: The
      automatic study state transition happens precisely as described above;
      that is, deleting trials or updating StudyConfig NEVER automatically
      moves the study state back to ACTIVE. If you want to _resume_ a Study
      that was stopped, 1) change the stopping conditions if necessary, 2)
      activate the study, and then 3) ask for suggestions. If the specified
      time or duration has not passed, do not stop the study.
    shouldStopAsap: If true, a Study enters STOPPING_ASAP whenever it would
      normally enters STOPPING state. The bottom line is: set to true if you
      want to interrupt on-going evaluations of Trials as soon as the study
      stopping condition is met. (Please see Study.State documentation for the
      source of truth).
  """
    maxDurationNoProgress = _messages.StringField(1)
    maxNumTrials = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    maxNumTrialsNoProgress = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    maximumRuntimeConstraint = _messages.MessageField('GoogleCloudAiplatformV1StudyTimeConstraint', 4)
    minNumTrials = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    minimumRuntimeConstraint = _messages.MessageField('GoogleCloudAiplatformV1StudyTimeConstraint', 6)
    shouldStopAsap = _messages.BooleanField(7)