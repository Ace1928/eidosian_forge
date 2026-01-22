from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Featurestore(_messages.Message):
    """Vertex AI Feature Store provides a centralized repository for
  organizing, storing, and serving ML features. The Featurestore is a top-
  level container for your features and their values.

  Enums:
    StateValueValuesEnum: Output only. State of the featurestore.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata to organize
      your Featurestore. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      Featurestore(System labels are excluded)." System reserved label keys
      are prefixed with "aiplatform.googleapis.com/" and are immutable.

  Fields:
    createTime: Output only. Timestamp when this Featurestore was created.
    encryptionSpec: Optional. Customer-managed encryption key spec for data
      storage. If set, both of the online and offline data storage will be
      secured by this key.
    etag: Optional. Used to perform consistent read-modify-write updates. If
      not set, a blind "overwrite" update happens.
    labels: Optional. The labels with user-defined metadata to organize your
      Featurestore. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      Featurestore(System labels are excluded)." System reserved label keys
      are prefixed with "aiplatform.googleapis.com/" and are immutable.
    name: Output only. Name of the Featurestore. Format:
      `projects/{project}/locations/{location}/featurestores/{featurestore}`
    onlineServingConfig: Optional. Config for online storage resources. The
      field should not co-exist with the field of
      `OnlineStoreReplicationConfig`. If both of it and
      OnlineStoreReplicationConfig are unset, the feature store will not have
      an online store and cannot be used for online serving.
    state: Output only. State of the featurestore.
    updateTime: Output only. Timestamp when this Featurestore was last
      updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the featurestore.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      STABLE: State when the featurestore configuration is not being updated
        and the fields reflect the current configuration of the featurestore.
        The featurestore is usable in this state.
      UPDATING: The state of the featurestore configuration when it is being
        updated. During an update, the fields reflect either the original
        configuration or the updated configuration of the featurestore. For
        example, `online_serving_config.fixed_node_count` can take minutes to
        update. While the update is in progress, the featurestore is in the
        UPDATING state, and the value of `fixed_node_count` can be the
        original value or the updated value, depending on the progress of the
        operation. Until the update completes, the actual number of nodes can
        still be the original value of `fixed_node_count`. The featurestore is
        still usable in this state.
    """
        STATE_UNSPECIFIED = 0
        STABLE = 1
        UPDATING = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata to organize your
    Featurestore. Label keys and values can be no longer than 64 characters
    (Unicode codepoints), can only contain lowercase letters, numeric
    characters, underscores and dashes. International characters are allowed.
    See https://goo.gl/xmQnxf for more information on and examples of labels.
    No more than 64 user labels can be associated with one Featurestore(System
    labels are excluded)." System reserved label keys are prefixed with
    "aiplatform.googleapis.com/" and are immutable.

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
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1EncryptionSpec', 2)
    etag = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    onlineServingConfig = _messages.MessageField('GoogleCloudAiplatformV1FeaturestoreOnlineServingConfig', 6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    updateTime = _messages.StringField(8)