from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1NasJob(_messages.Message):
    """Represents a Neural Architecture Search (NAS) job.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of the job.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize NasJobs.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. See
      https://goo.gl/xmQnxf for more information and examples of labels.

  Fields:
    createTime: Output only. Time when the NasJob was created.
    displayName: Required. The display name of the NasJob. The name can be up
      to 128 characters long and can consist of any UTF-8 characters.
    enableRestrictedImageTraining: Optional. Enable a separation of Custom
      model training and restricted image training for tenant project.
    encryptionSpec: Customer-managed encryption key options for a NasJob. If
      this is set, then all resources created by the NasJob will be encrypted
      with the provided encryption key.
    endTime: Output only. Time when the NasJob entered any of the following
      states: `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`,
      `JOB_STATE_CANCELLED`.
    error: Output only. Only populated when job's state is JOB_STATE_FAILED or
      JOB_STATE_CANCELLED.
    labels: The labels with user-defined metadata to organize NasJobs. Label
      keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. See
      https://goo.gl/xmQnxf for more information and examples of labels.
    name: Output only. Resource name of the NasJob.
    nasJobOutput: Output only. Output of the NasJob.
    nasJobSpec: Required. The specification of a NasJob.
    startTime: Output only. Time when the NasJob for the first time entered
      the `JOB_STATE_RUNNING` state.
    state: Output only. The detailed state of the job.
    updateTime: Output only. Time when the NasJob was most recently updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of the job.

    Values:
      JOB_STATE_UNSPECIFIED: The job state is unspecified.
      JOB_STATE_QUEUED: The job has been just created or resumed and
        processing has not yet begun.
      JOB_STATE_PENDING: The service is preparing to run the job.
      JOB_STATE_RUNNING: The job is in progress.
      JOB_STATE_SUCCEEDED: The job completed successfully.
      JOB_STATE_FAILED: The job failed.
      JOB_STATE_CANCELLING: The job is being cancelled. From this state the
        job may only go to either `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED` or
        `JOB_STATE_CANCELLED`.
      JOB_STATE_CANCELLED: The job has been cancelled.
      JOB_STATE_PAUSED: The job has been stopped, and can be resumed.
      JOB_STATE_EXPIRED: The job has expired.
      JOB_STATE_UPDATING: The job is being updated. Only jobs in the `RUNNING`
        state can be updated. After updating, the job goes back to the
        `RUNNING` state.
      JOB_STATE_PARTIALLY_SUCCEEDED: The job is partially succeeded, some
        results may be missing due to errors.
    """
        JOB_STATE_UNSPECIFIED = 0
        JOB_STATE_QUEUED = 1
        JOB_STATE_PENDING = 2
        JOB_STATE_RUNNING = 3
        JOB_STATE_SUCCEEDED = 4
        JOB_STATE_FAILED = 5
        JOB_STATE_CANCELLING = 6
        JOB_STATE_CANCELLED = 7
        JOB_STATE_PAUSED = 8
        JOB_STATE_EXPIRED = 9
        JOB_STATE_UPDATING = 10
        JOB_STATE_PARTIALLY_SUCCEEDED = 11

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize NasJobs. Label keys
    and values can be no longer than 64 characters (Unicode codepoints), can
    only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. See https://goo.gl/xmQnxf
    for more information and examples of labels.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    enableRestrictedImageTraining = _messages.BooleanField(3)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1EncryptionSpec', 4)
    endTime = _messages.StringField(5)
    error = _messages.MessageField('GoogleRpcStatus', 6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    nasJobOutput = _messages.MessageField('GoogleCloudAiplatformV1beta1NasJobOutput', 9)
    nasJobSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1NasJobSpec', 10)
    startTime = _messages.StringField(11)
    state = _messages.EnumField('StateValueValuesEnum', 12)
    updateTime = _messages.StringField(13)