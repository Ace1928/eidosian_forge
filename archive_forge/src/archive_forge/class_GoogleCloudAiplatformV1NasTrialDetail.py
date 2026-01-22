from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NasTrialDetail(_messages.Message):
    """Represents a NasTrial details along with its parameters. If there is a
  corresponding train NasTrial, the train NasTrial is also returned.

  Fields:
    name: Output only. Resource name of the NasTrialDetail.
    parameters: The parameters for the NasJob NasTrial.
    searchTrial: The requested search NasTrial.
    trainTrial: The train NasTrial corresponding to search_trial. Only
      populated if search_trial is used for training.
  """
    name = _messages.StringField(1)
    parameters = _messages.StringField(2)
    searchTrial = _messages.MessageField('GoogleCloudAiplatformV1NasTrial', 3)
    trainTrial = _messages.MessageField('GoogleCloudAiplatformV1NasTrial', 4)