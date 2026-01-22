from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1CompleteTrialRequest(_messages.Message):
    """The request message for the CompleteTrial service method.

  Fields:
    finalMeasurement: Optional. If provided, it will be used as the completed
      trial's final_measurement; Otherwise, the service will auto-select a
      previously reported measurement as the final-measurement
    infeasibleReason: Optional. A human readable reason why the trial was
      infeasible. This should only be provided if `trial_infeasible` is true.
    trialInfeasible: Optional. True if the trial cannot be run with the given
      Parameter, and final_measurement will be ignored.
  """
    finalMeasurement = _messages.MessageField('GoogleCloudMlV1Measurement', 1)
    infeasibleReason = _messages.StringField(2)
    trialInfeasible = _messages.BooleanField(3)