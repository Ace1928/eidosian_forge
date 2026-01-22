from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Annotation(_messages.Message):
    """Used to assign specific AnnotationSpec to a particular area of a
  DataItem or the whole part of the DataItem.

  Messages:
    LabelsValue: Optional. The labels with user-defined metadata to organize
      your Annotations. Label keys and values can be no longer than 64
      characters (Unicode codepoints), can only contain lowercase letters,
      numeric characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one
      Annotation(System labels are excluded). See https://goo.gl/xmQnxf for
      more information and examples of labels. System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable. Following
      system labels exist for each Annotation: *
      "aiplatform.googleapis.com/annotation_set_name": optional, name of the
      UI's annotation set this Annotation belongs to. If not set, the
      Annotation is not visible in the UI. *
      "aiplatform.googleapis.com/payload_schema": output only, its value is
      the payload_schema's title.

  Fields:
    annotationSource: Output only. The source of the Annotation.
    createTime: Output only. Timestamp when this Annotation was created.
    etag: Optional. Used to perform consistent read-modify-write updates. If
      not set, a blind "overwrite" update happens.
    labels: Optional. The labels with user-defined metadata to organize your
      Annotations. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one
      Annotation(System labels are excluded). See https://goo.gl/xmQnxf for
      more information and examples of labels. System reserved label keys are
      prefixed with "aiplatform.googleapis.com/" and are immutable. Following
      system labels exist for each Annotation: *
      "aiplatform.googleapis.com/annotation_set_name": optional, name of the
      UI's annotation set this Annotation belongs to. If not set, the
      Annotation is not visible in the UI. *
      "aiplatform.googleapis.com/payload_schema": output only, its value is
      the payload_schema's title.
    name: Output only. Resource name of the Annotation.
    payload: Required. The schema of the payload can be found in
      payload_schema.
    payloadSchemaUri: Required. Google Cloud Storage URI points to a YAML file
      describing payload. The schema is defined as an [OpenAPI 3.0.2 Schema
      Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject). The schema
      files that can be used here are found in gs://google-cloud-
      aiplatform/schema/dataset/annotation/, note that the chosen schema must
      be consistent with the parent Dataset's metadata.
    updateTime: Output only. Timestamp when this Annotation was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels with user-defined metadata to organize your
    Annotations. Label keys and values can be no longer than 64 characters
    (Unicode codepoints), can only contain lowercase letters, numeric
    characters, underscores and dashes. International characters are allowed.
    No more than 64 user labels can be associated with one Annotation(System
    labels are excluded). See https://goo.gl/xmQnxf for more information and
    examples of labels. System reserved label keys are prefixed with
    "aiplatform.googleapis.com/" and are immutable. Following system labels
    exist for each Annotation: *
    "aiplatform.googleapis.com/annotation_set_name": optional, name of the
    UI's annotation set this Annotation belongs to. If not set, the Annotation
    is not visible in the UI. * "aiplatform.googleapis.com/payload_schema":
    output only, its value is the payload_schema's title.

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
    annotationSource = _messages.MessageField('GoogleCloudAiplatformV1UserActionReference', 1)
    createTime = _messages.StringField(2)
    etag = _messages.StringField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    payload = _messages.MessageField('extra_types.JsonValue', 6)
    payloadSchemaUri = _messages.StringField(7)
    updateTime = _messages.StringField(8)