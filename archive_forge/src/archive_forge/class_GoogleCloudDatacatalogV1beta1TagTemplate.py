from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1TagTemplate(_messages.Message):
    """A tag template defines a tag, which can have one or more typed fields.
  The template is used to create and attach the tag to Google Cloud resources.
  [Tag template roles](https://cloud.google.com/iam/docs/understanding-
  roles#data-catalog-roles) provide permissions to create, edit, and use the
  template. See, for example, the [TagTemplate
  User](https://cloud.google.com/data-catalog/docs/how-to/template-user) role,
  which includes permission to use the tag template to tag resources.

  Enums:
    DataplexTransferStatusValueValuesEnum: Output only. Transfer status of the
      TagTemplate

  Messages:
    FieldsValue: Required. Map of tag template field IDs to the settings for
      the field. This map is an exhaustive list of the allowed fields. This
      map must contain at least one field and at most 500 fields. The keys to
      this map are tag template field IDs. Field IDs can contain letters (both
      uppercase and lowercase), numbers (0-9) and underscores (_). Field IDs
      must be at least 1 character long and at most 64 characters long. Field
      IDs must start with a letter or underscore.

  Fields:
    dataplexTransferStatus: Output only. Transfer status of the TagTemplate
    displayName: The display name for this template. Defaults to an empty
      string.
    fields: Required. Map of tag template field IDs to the settings for the
      field. This map is an exhaustive list of the allowed fields. This map
      must contain at least one field and at most 500 fields. The keys to this
      map are tag template field IDs. Field IDs can contain letters (both
      uppercase and lowercase), numbers (0-9) and underscores (_). Field IDs
      must be at least 1 character long and at most 64 characters long. Field
      IDs must start with a letter or underscore.
    name: Identifier. The resource name of the tag template in URL format.
      Example: * projects/{project_id}/locations/{location}/tagTemplates/{tag_
      template_id} Note that this TagTemplate and its child resources may not
      actually be stored in the location in this name.
  """

    class DataplexTransferStatusValueValuesEnum(_messages.Enum):
        """Output only. Transfer status of the TagTemplate

    Values:
      DATAPLEX_TRANSFER_STATUS_UNSPECIFIED: Default value. TagTemplate and its
        tags are only visible and editable in DataCatalog.
      MIGRATED: TagTemplate and its tags are auto-copied to Dataplex service.
        Visible in both services. Editable in DataCatalog, read-only in
        Dataplex.
    """
        DATAPLEX_TRANSFER_STATUS_UNSPECIFIED = 0
        MIGRATED = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FieldsValue(_messages.Message):
        """Required. Map of tag template field IDs to the settings for the field.
    This map is an exhaustive list of the allowed fields. This map must
    contain at least one field and at most 500 fields. The keys to this map
    are tag template field IDs. Field IDs can contain letters (both uppercase
    and lowercase), numbers (0-9) and underscores (_). Field IDs must be at
    least 1 character long and at most 64 characters long. Field IDs must
    start with a letter or underscore.

    Messages:
      AdditionalProperty: An additional property for a FieldsValue object.

    Fields:
      additionalProperties: Additional properties of type FieldsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FieldsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDatacatalogV1beta1TagTemplateField attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDatacatalogV1beta1TagTemplateField', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    dataplexTransferStatus = _messages.EnumField('DataplexTransferStatusValueValuesEnum', 1)
    displayName = _messages.StringField(2)
    fields = _messages.MessageField('FieldsValue', 3)
    name = _messages.StringField(4)