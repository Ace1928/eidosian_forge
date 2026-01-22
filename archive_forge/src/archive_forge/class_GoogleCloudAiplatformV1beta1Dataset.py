from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Dataset(_messages.Message):
    """A collection of DataItems and Annotations on them.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      Datasets. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one Dataset
      (System labels are excluded). See https://goo.gl/xmQnxf for more
      information and examples of labels. System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable. Following
      system labels exist for each Dataset: *
      "aiplatform.googleapis.com/dataset_metadata_schema": output only, its
      value is the metadata_schema's title.

  Fields:
    createTime: Output only. Timestamp when this Dataset was created.
    dataItemCount: Output only. The number of DataItems in this Dataset. Only
      apply for non-structured Dataset.
    description: The description of the Dataset.
    displayName: Required. The user-defined name of the Dataset. The name can
      be up to 128 characters long and can consist of any UTF-8 characters.
    encryptionSpec: Customer-managed encryption key spec for a Dataset. If
      set, this Dataset and all sub-resources of this Dataset will be secured
      by this key.
    etag: Used to perform consistent read-modify-write updates. If not set, a
      blind "overwrite" update happens.
    labels: The labels with user-defined metadata to organize your Datasets.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. No more
      than 64 user labels can be associated with one Dataset (System labels
      are excluded). See https://goo.gl/xmQnxf for more information and
      examples of labels. System reserved label keys are prefixed with
      "aiplatform.googleapis.com/" and are immutable. Following system labels
      exist for each Dataset: *
      "aiplatform.googleapis.com/dataset_metadata_schema": output only, its
      value is the metadata_schema's title.
    metadata: Required. Additional information about the Dataset.
    metadataArtifact: Output only. The resource name of the Artifact that was
      created in MetadataStore when creating the Dataset. The Artifact
      resource name pattern is `projects/{project}/locations/{location}/metada
      taStores/{metadata_store}/artifacts/{artifact}`.
    metadataSchemaUri: Required. Points to a YAML file stored on Google Cloud
      Storage describing additional information about the Dataset. The schema
      is defined as an OpenAPI 3.0.2 Schema Object. The schema files that can
      be used here are found in gs://google-cloud-
      aiplatform/schema/dataset/metadata/.
    name: Output only. The resource name of the Dataset.
    savedQueries: All SavedQueries belong to the Dataset will be returned in
      List/Get Dataset response. The annotation_specs field will not be
      populated except for UI cases which will only use annotation_spec_count.
      In CreateDataset request, a SavedQuery is created together if this field
      is set, up to one SavedQuery can be set in CreateDatasetRequest. The
      SavedQuery should not contain any AnnotationSpec.
    updateTime: Output only. Timestamp when this Dataset was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your Datasets. Label
    keys and values can be no longer than 64 characters (Unicode codepoints),
    can only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. No more than 64 user labels
    can be associated with one Dataset (System labels are excluded). See
    https://goo.gl/xmQnxf for more information and examples of labels. System
    reserved label keys are prefixed with "aiplatform.googleapis.com/" and are
    immutable. Following system labels exist for each Dataset: *
    "aiplatform.googleapis.com/dataset_metadata_schema": output only, its
    value is the metadata_schema's title.

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
    dataItemCount = _messages.IntegerField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1EncryptionSpec', 5)
    etag = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    metadata = _messages.MessageField('extra_types.JsonValue', 8)
    metadataArtifact = _messages.StringField(9)
    metadataSchemaUri = _messages.StringField(10)
    name = _messages.StringField(11)
    savedQueries = _messages.MessageField('GoogleCloudAiplatformV1beta1SavedQuery', 12, repeated=True)
    updateTime = _messages.StringField(13)