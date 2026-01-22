from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1TrainingConfig(_messages.Message):
    """CMLE training config. For every active learning labeling iteration,
  system will train a machine learning model on CMLE. The trained model will
  be used by data sampling algorithm to select DataItems.

  Fields:
    timeoutTrainingMilliHours: The timeout hours for the CMLE training job,
      expressed in milli hours i.e. 1,000 value in this field means 1 hour.
  """
    timeoutTrainingMilliHours = _messages.IntegerField(1)