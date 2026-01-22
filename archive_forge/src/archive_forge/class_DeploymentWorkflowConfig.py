from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentWorkflowConfig(_messages.Message):
    """Dataflow job deployment workflow config.

  Enums:
    JobTypeValueValuesEnum: Streaming or batch job.
    StopModeValueValuesEnum: Stop modes for the existent job.

  Fields:
    jobType: Streaming or batch job.
    serviceAccount: Custom service account to use for the deployment. By
      default the service account provisioned for CICD will be used.
    snapshotPolicy: Snapshot policies to take a snapshot of the existing
      dataflow job before beginning the deployment.
    stopMode: Stop modes for the existent job.
  """

    class JobTypeValueValuesEnum(_messages.Enum):
        """Streaming or batch job.

    Values:
      JOB_TYPE_UNSPECIFIED: Job type is unspecified.
      JOB_TYPE_BATCH: Batch job.
      JOB_TYPE_STREAMING: Streaming job.
    """
        JOB_TYPE_UNSPECIFIED = 0
        JOB_TYPE_BATCH = 1
        JOB_TYPE_STREAMING = 2

    class StopModeValueValuesEnum(_messages.Enum):
        """Stop modes for the existent job.

    Values:
      STOP_MODE_UNSPECIFIED: Stop mode is unspecified.
      STOP_MODE_DRAIN: Drain the job.
      STOP_MODE_CANCEL: Cancel the job.
    """
        STOP_MODE_UNSPECIFIED = 0
        STOP_MODE_DRAIN = 1
        STOP_MODE_CANCEL = 2
    jobType = _messages.EnumField('JobTypeValueValuesEnum', 1)
    serviceAccount = _messages.StringField(2)
    snapshotPolicy = _messages.MessageField('SnapshotPolicy', 3)
    stopMode = _messages.EnumField('StopModeValueValuesEnum', 4)