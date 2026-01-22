from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1Tag(_messages.Message):
    """Tags contain custom metadata and are attached to Data Catalog resources.
  Tags conform with the specification of their tag template. See [Data Catalog
  IAM](https://cloud.google.com/data-catalog/docs/concepts/iam) for
  information on the permissions needed to create or view tags.

  Messages:
    FieldsValue: Required. Maps the ID of a tag field to its value and
      additional information about that field. Tag template defines valid
      field IDs. A tag must have at least 1 field and at most 500 fields.

  Fields:
    column: Resources like entry can have schemas associated with them. This
      scope allows you to attach tags to an individual column based on that
      schema. To attach a tag to a nested column, separate column names with a
      dot (`.`). Example: `column.nested_column`.
    fields: Required. Maps the ID of a tag field to its value and additional
      information about that field. Tag template defines valid field IDs. A
      tag must have at least 1 field and at most 500 fields.
    name: Identifier. The resource name of the tag in URL format where tag ID
      is a system-generated identifier. Note: The tag itself might not be
      stored in the location specified in its name.
    template: Required. The resource name of the tag template this tag uses.
      Example: `projects/{PROJECT_ID}/locations/{LOCATION}/tagTemplates/{TAG_T
      EMPLATE_ID}` This field cannot be modified after creation.
    templateDisplayName: Output only. The display name of the tag template.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FieldsValue(_messages.Message):
        """Required. Maps the ID of a tag field to its value and additional
    information about that field. Tag template defines valid field IDs. A tag
    must have at least 1 field and at most 500 fields.

    Messages:
      AdditionalProperty: An additional property for a FieldsValue object.

    Fields:
      additionalProperties: Additional properties of type FieldsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FieldsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDatacatalogV1TagField attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDatacatalogV1TagField', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    column = _messages.StringField(1)
    fields = _messages.MessageField('FieldsValue', 2)
    name = _messages.StringField(3)
    template = _messages.StringField(4)
    templateDisplayName = _messages.StringField(5)