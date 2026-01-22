from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NasJobOutput(_messages.Message):
    """Represents a uCAIP NasJob output.

  Fields:
    multiTrialJobOutput: Output only. The output of this multi-trial Neural
      Architecture Search (NAS) job.
  """
    multiTrialJobOutput = _messages.MessageField('GoogleCloudAiplatformV1NasJobOutputMultiTrialJobOutput', 1)