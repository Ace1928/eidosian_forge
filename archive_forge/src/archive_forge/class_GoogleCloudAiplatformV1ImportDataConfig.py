from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ImportDataConfig(_messages.Message):
    """Describes the location from where we import data into a Dataset,
  together with the labels that will be applied to the DataItems and the
  Annotations.

  Messages:
    AnnotationLabelsValue: Labels that will be applied to newly imported
      Annotations. If two Annotations are identical, one of them will be
      deduped. Two Annotations are considered identical if their payload,
      payload_schema_uri and all of their labels are the same. These labels
      will be overridden by Annotation labels specified inside index file
      referenced by import_schema_uri, e.g. jsonl file.
    DataItemLabelsValue: Labels that will be applied to newly imported
      DataItems. If an identical DataItem as one being imported already exists
      in the Dataset, then these labels will be appended to these of the
      already existing one, and if labels with identical key is imported
      before, the old label value will be overwritten. If two DataItems are
      identical in the same import data operation, the labels will be combined
      and if key collision happens in this case, one of the values will be
      picked randomly. Two DataItems are considered identical if their content
      bytes are identical (e.g. image bytes or pdf bytes). These labels will
      be overridden by Annotation labels specified inside index file
      referenced by import_schema_uri, e.g. jsonl file.

  Fields:
    annotationLabels: Labels that will be applied to newly imported
      Annotations. If two Annotations are identical, one of them will be
      deduped. Two Annotations are considered identical if their payload,
      payload_schema_uri and all of their labels are the same. These labels
      will be overridden by Annotation labels specified inside index file
      referenced by import_schema_uri, e.g. jsonl file.
    dataItemLabels: Labels that will be applied to newly imported DataItems.
      If an identical DataItem as one being imported already exists in the
      Dataset, then these labels will be appended to these of the already
      existing one, and if labels with identical key is imported before, the
      old label value will be overwritten. If two DataItems are identical in
      the same import data operation, the labels will be combined and if key
      collision happens in this case, one of the values will be picked
      randomly. Two DataItems are considered identical if their content bytes
      are identical (e.g. image bytes or pdf bytes). These labels will be
      overridden by Annotation labels specified inside index file referenced
      by import_schema_uri, e.g. jsonl file.
    gcsSource: The Google Cloud Storage location for the input content.
    importSchemaUri: Required. Points to a YAML file stored on Google Cloud
      Storage describing the import format. Validation will be done against
      the schema. The schema is defined as an [OpenAPI 3.0.2 Schema
      Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject).
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationLabelsValue(_messages.Message):
        """Labels that will be applied to newly imported Annotations. If two
    Annotations are identical, one of them will be deduped. Two Annotations
    are considered identical if their payload, payload_schema_uri and all of
    their labels are the same. These labels will be overridden by Annotation
    labels specified inside index file referenced by import_schema_uri, e.g.
    jsonl file.

    Messages:
      AdditionalProperty: An additional property for a AnnotationLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        AnnotationLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DataItemLabelsValue(_messages.Message):
        """Labels that will be applied to newly imported DataItems. If an
    identical DataItem as one being imported already exists in the Dataset,
    then these labels will be appended to these of the already existing one,
    and if labels with identical key is imported before, the old label value
    will be overwritten. If two DataItems are identical in the same import
    data operation, the labels will be combined and if key collision happens
    in this case, one of the values will be picked randomly. Two DataItems are
    considered identical if their content bytes are identical (e.g. image
    bytes or pdf bytes). These labels will be overridden by Annotation labels
    specified inside index file referenced by import_schema_uri, e.g. jsonl
    file.

    Messages:
      AdditionalProperty: An additional property for a DataItemLabelsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DataItemLabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DataItemLabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotationLabels = _messages.MessageField('AnnotationLabelsValue', 1)
    dataItemLabels = _messages.MessageField('DataItemLabelsValue', 2)
    gcsSource = _messages.MessageField('GoogleCloudAiplatformV1GcsSource', 3)
    importSchemaUri = _messages.StringField(4)