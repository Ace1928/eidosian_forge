from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessSparkApplicationStageAttemptResponse(_messages.Message):
    """Stage Attempt for a Stage of a Spark Application

  Fields:
    stageData: Output only. Data corresponding to a stage.
  """
    stageData = _messages.MessageField('StageData', 1)