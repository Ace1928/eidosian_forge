from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PublisherModel(_messages.Message):
    """A Model Garden Publisher Model.

  Enums:
    LaunchStageValueValuesEnum: Optional. Indicates the launch stage of the
      model.
    OpenSourceCategoryValueValuesEnum: Required. Indicates the open source
      category of the publisher model.
    VersionStateValueValuesEnum: Optional. Indicates the state of the model
      version.

  Fields:
    frameworks: Optional. Additional information about the model's Frameworks.
    launchStage: Optional. Indicates the launch stage of the model.
    name: Output only. The resource name of the PublisherModel.
    openSourceCategory: Required. Indicates the open source category of the
      publisher model.
    parent: Optional. The parent that this model was customized from. E.g.,
      Vision API, Natural Language API, LaMDA, T5, etc. Foundation models
      don't have parents.
    predictSchemata: Optional. The schemata that describes formats of the
      PublisherModel's predictions and explanations as given and returned via
      PredictionService.Predict.
    publisherModelTemplate: Optional. Output only. Immutable. Used to indicate
      this model has a publisher model and provide the template of the
      publisher model resource name.
    supportedActions: Optional. Supported call-to-action options.
    versionId: Output only. Immutable. The version ID of the PublisherModel. A
      new version is committed when a new model version is uploaded under an
      existing model id. It is an auto-incrementing decimal number in string
      representation.
    versionState: Optional. Indicates the state of the model version.
  """

    class LaunchStageValueValuesEnum(_messages.Enum):
        """Optional. Indicates the launch stage of the model.

    Values:
      LAUNCH_STAGE_UNSPECIFIED: The model launch stage is unspecified.
      EXPERIMENTAL: Used to indicate the PublisherModel is at Experimental
        launch stage, available to a small set of customers.
      PRIVATE_PREVIEW: Used to indicate the PublisherModel is at Private
        Preview launch stage, only available to a small set of customers,
        although a larger set of customers than an Experimental launch.
        Previews are the first launch stage used to get feedback from
        customers.
      PUBLIC_PREVIEW: Used to indicate the PublisherModel is at Public Preview
        launch stage, available to all customers, although not supported for
        production workloads.
      GA: Used to indicate the PublisherModel is at GA launch stage, available
        to all customers and ready for production workload.
    """
        LAUNCH_STAGE_UNSPECIFIED = 0
        EXPERIMENTAL = 1
        PRIVATE_PREVIEW = 2
        PUBLIC_PREVIEW = 3
        GA = 4

    class OpenSourceCategoryValueValuesEnum(_messages.Enum):
        """Required. Indicates the open source category of the publisher model.

    Values:
      OPEN_SOURCE_CATEGORY_UNSPECIFIED: The open source category is
        unspecified, which should not be used.
      PROPRIETARY: Used to indicate the PublisherModel is not open sourced.
      GOOGLE_OWNED_OSS_WITH_GOOGLE_CHECKPOINT: Used to indicate the
        PublisherModel is a Google-owned open source model w/ Google
        checkpoint.
      THIRD_PARTY_OWNED_OSS_WITH_GOOGLE_CHECKPOINT: Used to indicate the
        PublisherModel is a 3p-owned open source model w/ Google checkpoint.
      GOOGLE_OWNED_OSS: Used to indicate the PublisherModel is a Google-owned
        pure open source model.
      THIRD_PARTY_OWNED_OSS: Used to indicate the PublisherModel is a 3p-owned
        pure open source model.
    """
        OPEN_SOURCE_CATEGORY_UNSPECIFIED = 0
        PROPRIETARY = 1
        GOOGLE_OWNED_OSS_WITH_GOOGLE_CHECKPOINT = 2
        THIRD_PARTY_OWNED_OSS_WITH_GOOGLE_CHECKPOINT = 3
        GOOGLE_OWNED_OSS = 4
        THIRD_PARTY_OWNED_OSS = 5

    class VersionStateValueValuesEnum(_messages.Enum):
        """Optional. Indicates the state of the model version.

    Values:
      VERSION_STATE_UNSPECIFIED: The version state is unspecified.
      VERSION_STATE_STABLE: Used to indicate the version is stable.
      VERSION_STATE_UNSTABLE: Used to indicate the version is unstable.
    """
        VERSION_STATE_UNSPECIFIED = 0
        VERSION_STATE_STABLE = 1
        VERSION_STATE_UNSTABLE = 2
    frameworks = _messages.StringField(1, repeated=True)
    launchStage = _messages.EnumField('LaunchStageValueValuesEnum', 2)
    name = _messages.StringField(3)
    openSourceCategory = _messages.EnumField('OpenSourceCategoryValueValuesEnum', 4)
    parent = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelParent', 5)
    predictSchemata = _messages.MessageField('GoogleCloudAiplatformV1beta1PredictSchemata', 6)
    publisherModelTemplate = _messages.StringField(7)
    supportedActions = _messages.MessageField('GoogleCloudAiplatformV1beta1PublisherModelCallToAction', 8)
    versionId = _messages.StringField(9)
    versionState = _messages.EnumField('VersionStateValueValuesEnum', 10)