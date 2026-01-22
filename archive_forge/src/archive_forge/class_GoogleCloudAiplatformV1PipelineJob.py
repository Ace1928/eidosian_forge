from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PipelineJob(_messages.Message):
    """An instance of a machine learning PipelineJob.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of the job.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize
      PipelineJob. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels. Note there is some reserved label key for Vertex AI Pipelines. -
      `vertex-ai-pipelines-run-billing-id`, user set value will get overrided.
    PipelineSpecValue: The spec of the pipeline.

  Fields:
    createTime: Output only. Pipeline creation time.
    displayName: The display name of the Pipeline. The name can be up to 128
      characters long and can consist of any UTF-8 characters.
    encryptionSpec: Customer-managed encryption key spec for a pipelineJob. If
      set, this PipelineJob and all of its sub-resources will be secured by
      this key.
    endTime: Output only. Pipeline end time.
    error: Output only. The error that occurred during pipeline execution.
      Only populated when the pipeline's state is FAILED or CANCELLED.
    jobDetail: Output only. The details of pipeline run. Not available in the
      list view.
    labels: The labels with user-defined metadata to organize PipelineJob.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. See
      https://goo.gl/xmQnxf for more information and examples of labels. Note
      there is some reserved label key for Vertex AI Pipelines. - `vertex-ai-
      pipelines-run-billing-id`, user set value will get overrided.
    name: Output only. The resource name of the PipelineJob.
    network: The full name of the Compute Engine
      [network](/compute/docs/networks-and-firewalls#networks) to which the
      Pipeline Job's workload should be peered. For example,
      `projects/12345/global/networks/myVPC`.
      [Format](/compute/docs/reference/rest/v1/networks/insert) is of the form
      `projects/{project}/global/networks/{network}`. Where {project} is a
      project number, as in `12345`, and {network} is a network name. Private
      services access must already be configured for the network. Pipeline job
      will apply the network configuration to the Google Cloud resources being
      launched, if applied, such as Vertex AI Training or Dataflow job. If
      left unspecified, the workload is not peered with any network.
    pipelineSpec: The spec of the pipeline.
    runtimeConfig: Runtime config of the pipeline.
    scheduleName: Output only. The schedule resource name. Only returned if
      the Pipeline is created by Schedule API.
    serviceAccount: The service account that the pipeline workload runs as. If
      not specified, the Compute Engine default service account in the project
      will be used. See https://cloud.google.com/compute/docs/access/service-
      accounts#default_service_account Users starting the pipeline must have
      the `iam.serviceAccounts.actAs` permission on this service account.
    startTime: Output only. Pipeline start time.
    state: Output only. The detailed state of the job.
    templateMetadata: Output only. Pipeline template metadata. Will fill up
      fields if PipelineJob.template_uri is from supported template registry.
    templateUri: A template uri from where the PipelineJob.pipeline_spec, if
      empty, will be downloaded. Currently, only uri from Vertex Template
      Registry & Gallery is supported. Reference to
      https://cloud.google.com/vertex-ai/docs/pipelines/create-pipeline-
      template.
    updateTime: Output only. Timestamp when this PipelineJob was most recently
      updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of the job.

    Values:
      PIPELINE_STATE_UNSPECIFIED: The pipeline state is unspecified.
      PIPELINE_STATE_QUEUED: The pipeline has been created or resumed, and
        processing has not yet begun.
      PIPELINE_STATE_PENDING: The service is preparing to run the pipeline.
      PIPELINE_STATE_RUNNING: The pipeline is in progress.
      PIPELINE_STATE_SUCCEEDED: The pipeline completed successfully.
      PIPELINE_STATE_FAILED: The pipeline failed.
      PIPELINE_STATE_CANCELLING: The pipeline is being cancelled. From this
        state, the pipeline may only go to either PIPELINE_STATE_SUCCEEDED,
        PIPELINE_STATE_FAILED or PIPELINE_STATE_CANCELLED.
      PIPELINE_STATE_CANCELLED: The pipeline has been cancelled.
      PIPELINE_STATE_PAUSED: The pipeline has been stopped, and can be
        resumed.
    """
        PIPELINE_STATE_UNSPECIFIED = 0
        PIPELINE_STATE_QUEUED = 1
        PIPELINE_STATE_PENDING = 2
        PIPELINE_STATE_RUNNING = 3
        PIPELINE_STATE_SUCCEEDED = 4
        PIPELINE_STATE_FAILED = 5
        PIPELINE_STATE_CANCELLING = 6
        PIPELINE_STATE_CANCELLED = 7
        PIPELINE_STATE_PAUSED = 8

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize PipelineJob. Label
    keys and values can be no longer than 64 characters (Unicode codepoints),
    can only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. See https://goo.gl/xmQnxf
    for more information and examples of labels. Note there is some reserved
    label key for Vertex AI Pipelines. - `vertex-ai-pipelines-run-billing-id`,
    user set value will get overrided.

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
    class PipelineSpecValue(_messages.Message):
        """The spec of the pipeline.

    Messages:
      AdditionalProperty: An additional property for a PipelineSpecValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PipelineSpecValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    displayName = _messages.StringField(2)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1EncryptionSpec', 3)
    endTime = _messages.StringField(4)
    error = _messages.MessageField('GoogleRpcStatus', 5)
    jobDetail = _messages.MessageField('GoogleCloudAiplatformV1PipelineJobDetail', 6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    network = _messages.StringField(9)
    pipelineSpec = _messages.MessageField('PipelineSpecValue', 10)
    runtimeConfig = _messages.MessageField('GoogleCloudAiplatformV1PipelineJobRuntimeConfig', 11)
    scheduleName = _messages.StringField(12)
    serviceAccount = _messages.StringField(13)
    startTime = _messages.StringField(14)
    state = _messages.EnumField('StateValueValuesEnum', 15)
    templateMetadata = _messages.MessageField('GoogleCloudAiplatformV1PipelineTemplateMetadata', 16)
    templateUri = _messages.StringField(17)
    updateTime = _messages.StringField(18)