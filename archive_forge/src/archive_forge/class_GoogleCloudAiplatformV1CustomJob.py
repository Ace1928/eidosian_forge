from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1CustomJob(_messages.Message):
    """Represents a job that runs custom workloads such as a Docker container
  or a Python package. A CustomJob can have multiple worker pools and each
  worker pool can have its own machine and input spec. A CustomJob will be
  cleaned up once the job enters terminal state (failed or succeeded).

  Enums:
    StateValueValuesEnum: Output only. The detailed state of the job.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize CustomJobs.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. See
      https://goo.gl/xmQnxf for more information and examples of labels.
    WebAccessUrisValue: Output only. URIs for accessing [interactive
      shells](https://cloud.google.com/vertex-ai/docs/training/monitor-debug-
      interactive-shell) (one URI for each training node). Only available if
      job_spec.enable_web_access is `true`. The keys are names of each node in
      the training job; for example, `workerpool0-0` for the primary node,
      `workerpool1-0` for the first node in the second worker pool, and
      `workerpool1-1` for the second node in the second worker pool. The
      values are the URIs for each node's interactive shell.

  Fields:
    createTime: Output only. Time when the CustomJob was created.
    displayName: Required. The display name of the CustomJob. The name can be
      up to 128 characters long and can consist of any UTF-8 characters.
    encryptionSpec: Customer-managed encryption key options for a CustomJob.
      If this is set, then all resources created by the CustomJob will be
      encrypted with the provided encryption key.
    endTime: Output only. Time when the CustomJob entered any of the following
      states: `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`,
      `JOB_STATE_CANCELLED`.
    error: Output only. Only populated when job's state is `JOB_STATE_FAILED`
      or `JOB_STATE_CANCELLED`.
    jobSpec: Required. Job spec.
    labels: The labels with user-defined metadata to organize CustomJobs.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. See
      https://goo.gl/xmQnxf for more information and examples of labels.
    name: Output only. Resource name of a CustomJob.
    startTime: Output only. Time when the CustomJob for the first time entered
      the `JOB_STATE_RUNNING` state.
    state: Output only. The detailed state of the job.
    updateTime: Output only. Time when the CustomJob was most recently
      updated.
    webAccessUris: Output only. URIs for accessing [interactive
      shells](https://cloud.google.com/vertex-ai/docs/training/monitor-debug-
      interactive-shell) (one URI for each training node). Only available if
      job_spec.enable_web_access is `true`. The keys are names of each node in
      the training job; for example, `workerpool0-0` for the primary node,
      `workerpool1-0` for the first node in the second worker pool, and
      `workerpool1-1` for the second node in the second worker pool. The
      values are the URIs for each node's interactive shell.
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
        """The labels with user-defined metadata to organize CustomJobs. Label
    keys and values can be no longer than 64 characters (Unicode codepoints),
    can only contain lowercase letters, numeric characters, underscores and
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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class WebAccessUrisValue(_messages.Message):
        """Output only. URIs for accessing [interactive
    shells](https://cloud.google.com/vertex-ai/docs/training/monitor-debug-
    interactive-shell) (one URI for each training node). Only available if
    job_spec.enable_web_access is `true`. The keys are names of each node in
    the training job; for example, `workerpool0-0` for the primary node,
    `workerpool1-0` for the first node in the second worker pool, and
    `workerpool1-1` for the second node in the second worker pool. The values
    are the URIs for each node's interactive shell.

    Messages:
      AdditionalProperty: An additional property for a WebAccessUrisValue
        object.

    Fields:
      additionalProperties: Additional properties of type WebAccessUrisValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a WebAccessUrisValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1EncryptionSpec', 3)
    endTime = _messages.StringField(4)
    error = _messages.MessageField('GoogleRpcStatus', 5)
    jobSpec = _messages.MessageField('GoogleCloudAiplatformV1CustomJobSpec', 6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    startTime = _messages.StringField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    updateTime = _messages.StringField(11)
    webAccessUris = _messages.MessageField('WebAccessUrisValue', 12)