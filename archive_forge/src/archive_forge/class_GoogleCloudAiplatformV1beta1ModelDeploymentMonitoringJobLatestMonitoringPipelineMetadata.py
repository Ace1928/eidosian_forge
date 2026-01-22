from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelDeploymentMonitoringJobLatestMonitoringPipelineMetadata(_messages.Message):
    """All metadata of most recent monitoring pipelines.

  Fields:
    runTime: The time that most recent monitoring pipelines that is related to
      this run.
    status: The status of the most recent monitoring pipeline.
  """
    runTime = _messages.StringField(1)
    status = _messages.MessageField('GoogleRpcStatus', 2)