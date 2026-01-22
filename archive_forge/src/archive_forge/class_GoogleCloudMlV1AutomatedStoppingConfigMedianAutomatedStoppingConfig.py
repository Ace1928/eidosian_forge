from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1AutomatedStoppingConfigMedianAutomatedStoppingConfig(_messages.Message):
    """The median automated stopping rule stops a pending trial if the trial's
  best objective_value is strictly below the median 'performance' of all
  completed trials reported up to the trial's last measurement. Currently,
  'performance' refers to the running average of the objective values reported
  by the trial in each measurement.

  Fields:
    useElapsedTime: If true, the median automated stopping rule applies to
      measurement.use_elapsed_time, which means the elapsed_time field of the
      current trial's latest measurement is used to compute the median
      objective value for each completed trial.
  """
    useElapsedTime = _messages.BooleanField(1)