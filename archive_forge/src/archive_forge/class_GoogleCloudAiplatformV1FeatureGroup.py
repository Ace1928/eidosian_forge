from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureGroup(_messages.Message):
    """Vertex AI Feature Group.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata to organize
      your FeatureGroup. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      FeatureGroup(System labels are excluded)." System reserved label keys
      are prefixed with "aiplatform.googleapis.com/" and are immutable.

  Fields:
    bigQuery: Indicates that features for this group come from BigQuery
      Table/View. By default treats the source as a sparse time series source,
      which is required to have an entity_id and a feature_timestamp column in
      the source.
    createTime: Output only. Timestamp when this FeatureGroup was created.
    description: Optional. Description of the FeatureGroup.
    etag: Optional. Used to perform consistent read-modify-write updates. If
      not set, a blind "overwrite" update happens.
    labels: Optional. The labels with user-defined metadata to organize your
      FeatureGroup. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information on and examples
      of labels. No more than 64 user labels can be associated with one
      FeatureGroup(System labels are excluded)." System reserved label keys
      are prefixed with "aiplatform.googleapis.com/" and are immutable.
    name: Identifier. Name of the FeatureGroup. Format:
      `projects/{project}/locations/{location}/featureGroups/{featureGroup}`
    updateTime: Output only. Timestamp when this FeatureGroup was last
      updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata to organize your
    FeatureGroup. Label keys and values can be no longer than 64 characters
    (Unicode codepoints), can only contain lowercase letters, numeric
    characters, underscores and dashes. International characters are allowed.
    See https://goo.gl/xmQnxf for more information on and examples of labels.
    No more than 64 user labels can be associated with one FeatureGroup(System
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
    bigQuery = _messages.MessageField('GoogleCloudAiplatformV1FeatureGroupBigQuery', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    etag = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    updateTime = _messages.StringField(7)