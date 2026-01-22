from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1TrainingPipeline(_messages.Message):
    """The TrainingPipeline orchestrates tasks associated with training a
  Model. It always executes the training task, and optionally may also export
  data from Vertex AI's Dataset which becomes the training input, upload the
  Model to Vertex AI, and evaluate the Model.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of the pipeline.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize
      TrainingPipelines. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.

  Fields:
    createTime: Output only. Time when the TrainingPipeline was created.
    displayName: Required. The user-defined name of this TrainingPipeline.
    encryptionSpec: Customer-managed encryption key spec for a
      TrainingPipeline. If set, this TrainingPipeline will be secured by this
      key. Note: Model trained by this TrainingPipeline is also secured by
      this key if model_to_upload is not set separately.
    endTime: Output only. Time when the TrainingPipeline entered any of the
      following states: `PIPELINE_STATE_SUCCEEDED`, `PIPELINE_STATE_FAILED`,
      `PIPELINE_STATE_CANCELLED`.
    error: Output only. Only populated when the pipeline's state is
      `PIPELINE_STATE_FAILED` or `PIPELINE_STATE_CANCELLED`.
    inputDataConfig: Specifies Vertex AI owned input data that may be used for
      training the Model. The TrainingPipeline's training_task_definition
      should make clear whether this config is used and if there are any
      special requirements on how it should be filled. If nothing about this
      config is mentioned in the training_task_definition, then it should be
      assumed that the TrainingPipeline does not depend on this configuration.
    labels: The labels with user-defined metadata to organize
      TrainingPipelines. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.
    modelId: Optional. The ID to use for the uploaded Model, which will become
      the final component of the model resource name. This value may be up to
      63 characters, and valid characters are `[a-z0-9_-]`. The first
      character cannot be a number or hyphen.
    modelToUpload: Describes the Model that may be uploaded (via
      ModelService.UploadModel) by this TrainingPipeline. The
      TrainingPipeline's training_task_definition should make clear whether
      this Model description should be populated, and if there are any special
      requirements regarding how it should be filled. If nothing is mentioned
      in the training_task_definition, then it should be assumed that this
      field should not be filled and the training task either uploads the
      Model without a need of this information, or that training task does not
      support uploading a Model as part of the pipeline. When the Pipeline's
      state becomes `PIPELINE_STATE_SUCCEEDED` and the trained Model had been
      uploaded into Vertex AI, then the model_to_upload's resource name is
      populated. The Model is always uploaded into the Project and Location in
      which this pipeline is.
    name: Output only. Resource name of the TrainingPipeline.
    parentModel: Optional. When specify this field, the `model_to_upload` will
      not be uploaded as a new model, instead, it will become a new version of
      this `parent_model`.
    startTime: Output only. Time when the TrainingPipeline for the first time
      entered the `PIPELINE_STATE_RUNNING` state.
    state: Output only. The detailed state of the pipeline.
    trainingTaskDefinition: Required. A Google Cloud Storage path to the YAML
      file that defines the training task which is responsible for producing
      the model artifact, and may also include additional auxiliary work. The
      definition files that can be used here are found in gs://google-cloud-
      aiplatform/schema/trainingjob/definition/. Note: The URI given on output
      will be immutable and probably different, including the URI scheme, than
      the one given on input. The output URI will point to a location where
      the user only has a read access.
    trainingTaskInputs: Required. The training task's parameter(s), as
      specified in the training_task_definition's `inputs`.
    trainingTaskMetadata: Output only. The metadata information as specified
      in the training_task_definition's `metadata`. This metadata is an
      auxiliary runtime and final information about the training task. While
      the pipeline is running this information is populated only at a best
      effort basis. Only present if the pipeline's training_task_definition
      contains `metadata` object.
    updateTime: Output only. Time when the TrainingPipeline was most recently
      updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of the pipeline.

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
        """The labels with user-defined metadata to organize TrainingPipelines.
    Label keys and values can be no longer than 64 characters (Unicode
    codepoints), can only contain lowercase letters, numeric characters,
    underscores and dashes. International characters are allowed. See
    https://goo.gl/xmQnxf for more information and examples of labels.

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
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1EncryptionSpec', 3)
    endTime = _messages.StringField(4)
    error = _messages.MessageField('GoogleRpcStatus', 5)
    inputDataConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1InputDataConfig', 6)
    labels = _messages.MessageField('LabelsValue', 7)
    modelId = _messages.StringField(8)
    modelToUpload = _messages.MessageField('GoogleCloudAiplatformV1beta1Model', 9)
    name = _messages.StringField(10)
    parentModel = _messages.StringField(11)
    startTime = _messages.StringField(12)
    state = _messages.EnumField('StateValueValuesEnum', 13)
    trainingTaskDefinition = _messages.StringField(14)
    trainingTaskInputs = _messages.MessageField('extra_types.JsonValue', 15)
    trainingTaskMetadata = _messages.MessageField('extra_types.JsonValue', 16)
    updateTime = _messages.StringField(17)