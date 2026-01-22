from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1NasJobOutput(_messages.Message):
    """The output of Neural Archhitecture Search (NAS) jobs.

  Fields:
    multiTrialJobOutputs: The output of a multi-trial Neural Architecture
      Search (NAS) job.
  """
    multiTrialJobOutputs = _messages.MessageField('GoogleCloudMlV1NasJobOutputMultiTrialJobOutputs', 1)