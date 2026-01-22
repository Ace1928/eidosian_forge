from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureOnlineStore(_messages.Message):
    """Vertex AI Feature Online Store provides a centralized repository for
  serving ML features and embedding indexes at low latency. The Feature Online
  Store is a top-level container.

  Enums:
    StateValueValuesEnum: Output only. State of the featureOnlineStore.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata to organize
      your FeatureOnlineStore. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      FeatureOnlineStore(System labels are excluded)." System reserved label
      keys are prefixed with "aiplatform.googleapis.com/" and are immutable.

  Fields:
    bigtable: Contains settings for the Cloud Bigtable instance that will be
      created to serve featureValues for all FeatureViews under this
      FeatureOnlineStore.
    createTime: Output only. Timestamp when this FeatureOnlineStore was
      created.
    dedicatedServingEndpoint: Optional. The dedicated serving endpoint for
      this FeatureOnlineStore, which is different from common Vertex service
      endpoint.
    etag: Optional. Used to perform consistent read-modify-write updates. If
      not set, a blind "overwrite" update happens.
    labels: Optional. The labels with user-defined metadata to organize your
      FeatureOnlineStore. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      FeatureOnlineStore(System labels are excluded)." System reserved label
      keys are prefixed with "aiplatform.googleapis.com/" and are immutable.
    name: Identifier. Name of the FeatureOnlineStore. Format: `projects/{proje
      ct}/locations/{location}/featureOnlineStores/{featureOnlineStore}`
    optimized: Contains settings for the Optimized store that will be created
      to serve featureValues for all FeatureViews under this
      FeatureOnlineStore. When choose Optimized storage type, need to set
      PrivateServiceConnectConfig.enable_private_service_connect to use
      private endpoint. Otherwise will use public endpoint by default.
    state: Output only. State of the featureOnlineStore.
    updateTime: Output only. Timestamp when this FeatureOnlineStore was last
      updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the featureOnlineStore.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      STABLE: State when the featureOnlineStore configuration is not being
        updated and the fields reflect the current configuration of the
        featureOnlineStore. The featureOnlineStore is usable in this state.
      UPDATING: The state of the featureOnlineStore configuration when it is
        being updated. During an update, the fields reflect either the
        original configuration or the updated configuration of the
        featureOnlineStore. The featureOnlineStore is still usable in this
        state.
    """
        STATE_UNSPECIFIED = 0
        STABLE = 1
        UPDATING = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata to organize your
    FeatureOnlineStore. Label keys and values can be no longer than 64
    characters (Unicode codepoints), can only contain lowercase letters,
    numeric characters, underscores and dashes. International characters are
    allowed. See https://goo.gl/xmQnxf for more information on and examples of
    labels. No more than 64 user labels can be associated with one
    FeatureOnlineStore(System labels are excluded)." System reserved label
    keys are prefixed with "aiplatform.googleapis.com/" and are immutable.

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
    bigtable = _messages.MessageField('GoogleCloudAiplatformV1FeatureOnlineStoreBigtable', 1)
    createTime = _messages.StringField(2)
    dedicatedServingEndpoint = _messages.MessageField('GoogleCloudAiplatformV1FeatureOnlineStoreDedicatedServingEndpoint', 3)
    etag = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    optimized = _messages.MessageField('GoogleCloudAiplatformV1FeatureOnlineStoreOptimized', 7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    updateTime = _messages.StringField(9)