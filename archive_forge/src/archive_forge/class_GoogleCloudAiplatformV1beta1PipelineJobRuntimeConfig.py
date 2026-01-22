from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PipelineJobRuntimeConfig(_messages.Message):
    """The runtime config of a PipelineJob.

  Enums:
    FailurePolicyValueValuesEnum: Represents the failure policy of a pipeline.
      Currently, the default of a pipeline is that the pipeline will continue
      to run until no more tasks can be executed, also known as
      PIPELINE_FAILURE_POLICY_FAIL_SLOW. However, if a pipeline is set to
      PIPELINE_FAILURE_POLICY_FAIL_FAST, it will stop scheduling any new tasks
      when a task has failed. Any scheduled tasks will continue to completion.

  Messages:
    InputArtifactsValue: The runtime artifacts of the PipelineJob. The key
      will be the input artifact name and the value would be one of the
      InputArtifact.
    ParameterValuesValue: The runtime parameters of the PipelineJob. The
      parameters will be passed into PipelineJob.pipeline_spec to replace the
      placeholders at runtime. This field is used by pipelines built using
      `PipelineJob.pipeline_spec.schema_version` 2.1.0, such as pipelines
      built using Kubeflow Pipelines SDK 1.9 or higher and the v2 DSL.
    ParametersValue: Deprecated. Use RuntimeConfig.parameter_values instead.
      The runtime parameters of the PipelineJob. The parameters will be passed
      into PipelineJob.pipeline_spec to replace the placeholders at runtime.
      This field is used by pipelines built using
      `PipelineJob.pipeline_spec.schema_version` 2.0.0 or lower, such as
      pipelines built using Kubeflow Pipelines SDK 1.8 or lower.

  Fields:
    failurePolicy: Represents the failure policy of a pipeline. Currently, the
      default of a pipeline is that the pipeline will continue to run until no
      more tasks can be executed, also known as
      PIPELINE_FAILURE_POLICY_FAIL_SLOW. However, if a pipeline is set to
      PIPELINE_FAILURE_POLICY_FAIL_FAST, it will stop scheduling any new tasks
      when a task has failed. Any scheduled tasks will continue to completion.
    gcsOutputDirectory: Required. A path in a Cloud Storage bucket, which will
      be treated as the root output directory of the pipeline. It is used by
      the system to generate the paths of output artifacts. The artifact paths
      are generated with a sub-path pattern `{job_id}/{task_id}/{output_key}`
      under the specified output directory. The service account specified in
      this pipeline must have the `storage.objects.get` and
      `storage.objects.create` permissions for this bucket.
    inputArtifacts: The runtime artifacts of the PipelineJob. The key will be
      the input artifact name and the value would be one of the InputArtifact.
    parameterValues: The runtime parameters of the PipelineJob. The parameters
      will be passed into PipelineJob.pipeline_spec to replace the
      placeholders at runtime. This field is used by pipelines built using
      `PipelineJob.pipeline_spec.schema_version` 2.1.0, such as pipelines
      built using Kubeflow Pipelines SDK 1.9 or higher and the v2 DSL.
    parameters: Deprecated. Use RuntimeConfig.parameter_values instead. The
      runtime parameters of the PipelineJob. The parameters will be passed
      into PipelineJob.pipeline_spec to replace the placeholders at runtime.
      This field is used by pipelines built using
      `PipelineJob.pipeline_spec.schema_version` 2.0.0 or lower, such as
      pipelines built using Kubeflow Pipelines SDK 1.8 or lower.
  """

    class FailurePolicyValueValuesEnum(_messages.Enum):
        """Represents the failure policy of a pipeline. Currently, the default of
    a pipeline is that the pipeline will continue to run until no more tasks
    can be executed, also known as PIPELINE_FAILURE_POLICY_FAIL_SLOW. However,
    if a pipeline is set to PIPELINE_FAILURE_POLICY_FAIL_FAST, it will stop
    scheduling any new tasks when a task has failed. Any scheduled tasks will
    continue to completion.

    Values:
      PIPELINE_FAILURE_POLICY_UNSPECIFIED: Default value, and follows fail
        slow behavior.
      PIPELINE_FAILURE_POLICY_FAIL_SLOW: Indicates that the pipeline should
        continue to run until all possible tasks have been scheduled and
        completed.
      PIPELINE_FAILURE_POLICY_FAIL_FAST: Indicates that the pipeline should
        stop scheduling new tasks after a task has failed.
    """
        PIPELINE_FAILURE_POLICY_UNSPECIFIED = 0
        PIPELINE_FAILURE_POLICY_FAIL_SLOW = 1
        PIPELINE_FAILURE_POLICY_FAIL_FAST = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InputArtifactsValue(_messages.Message):
        """The runtime artifacts of the PipelineJob. The key will be the input
    artifact name and the value would be one of the InputArtifact.

    Messages:
      AdditionalProperty: An additional property for a InputArtifactsValue
        object.

    Fields:
      additionalProperties: Additional properties of type InputArtifactsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InputArtifactsValue object.

      Fields:
        key: Name of the additional property.
        value: A
          GoogleCloudAiplatformV1beta1PipelineJobRuntimeConfigInputArtifact
          attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1beta1PipelineJobRuntimeConfigInputArtifact', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParameterValuesValue(_messages.Message):
        """The runtime parameters of the PipelineJob. The parameters will be
    passed into PipelineJob.pipeline_spec to replace the placeholders at
    runtime. This field is used by pipelines built using
    `PipelineJob.pipeline_spec.schema_version` 2.1.0, such as pipelines built
    using Kubeflow Pipelines SDK 1.9 or higher and the v2 DSL.

    Messages:
      AdditionalProperty: An additional property for a ParameterValuesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ParameterValuesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParameterValuesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """Deprecated. Use RuntimeConfig.parameter_values instead. The runtime
    parameters of the PipelineJob. The parameters will be passed into
    PipelineJob.pipeline_spec to replace the placeholders at runtime. This
    field is used by pipelines built using
    `PipelineJob.pipeline_spec.schema_version` 2.0.0 or lower, such as
    pipelines built using Kubeflow Pipelines SDK 1.8 or lower.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Additional properties of type ParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudAiplatformV1beta1Value attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudAiplatformV1beta1Value', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    failurePolicy = _messages.EnumField('FailurePolicyValueValuesEnum', 1)
    gcsOutputDirectory = _messages.StringField(2)
    inputArtifacts = _messages.MessageField('InputArtifactsValue', 3)
    parameterValues = _messages.MessageField('ParameterValuesValue', 4)
    parameters = _messages.MessageField('ParametersValue', 5)