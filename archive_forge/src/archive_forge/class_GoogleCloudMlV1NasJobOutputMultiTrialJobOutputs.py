from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1NasJobOutputMultiTrialJobOutputs(_messages.Message):
    """The list of all MultiTrialJobOutput.

  Fields:
    multiTrialJobOutput: A GoogleCloudMlV1NasJobOutputMultiTrialJobOutput
      attribute.
  """
    multiTrialJobOutput = _messages.MessageField('GoogleCloudMlV1NasJobOutputMultiTrialJobOutput', 1, repeated=True)