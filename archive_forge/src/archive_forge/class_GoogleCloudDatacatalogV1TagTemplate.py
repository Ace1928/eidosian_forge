from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1TagTemplate(_messages.Message):
    """A tag template defines a tag that can have one or more typed fields. The
  template is used to create tags that are attached to Google Cloud resources.
  [Tag template roles] (https://cloud.google.com/iam/docs/understanding-
  roles#data-catalog-roles) provide permissions to create, edit, and use the
  template. For example, see the [TagTemplate User]
  (https://cloud.google.com/data-catalog/docs/how-to/template-user) role that
  includes a permission to use the tag template to tag resources.

  Enums:
    DataplexTransferStatusValueValuesEnum: Optional. Transfer status of the
      TagTemplate

  Messages:
    FieldsValue: Required. Map of tag template field IDs to the settings for
      the field. This map is an exhaustive list of the allowed fields. The map
      must contain at least one field and at most 500 fields. The keys to this
      map are tag template field IDs. The IDs have the following limitations:
      * Can contain uppercase and lowercase letters, numbers (0-9) and
      underscores (_). * Must be at least 1 character and at most 64
      characters long. * Must start with a letter or underscore.

  Fields:
    dataplexTransferStatus: Optional. Transfer status of the TagTemplate
    displayName: Display name for this template. Defaults to an empty string.
      The name must contain only Unicode letters, numbers (0-9), underscores
      (_), dashes (-), spaces ( ), and can't start or end with spaces. The
      maximum length is 200 characters.
    fields: Required. Map of tag template field IDs to the settings for the
      field. This map is an exhaustive list of the allowed fields. The map
      must contain at least one field and at most 500 fields. The keys to this
      map are tag template field IDs. The IDs have the following limitations:
      * Can contain uppercase and lowercase letters, numbers (0-9) and
      underscores (_). * Must be at least 1 character and at most 64
      characters long. * Must start with a letter or underscore.
    isPubliclyReadable: Indicates whether tags created with this template are
      public. Public tags do not require tag template access to appear in
      ListTags API response. Additionally, you can search for a public tag by
      value with a simple search query in addition to using a ``tag:``
      predicate.
    name: Identifier. The resource name of the tag template in URL format.
      Note: The tag template itself and its child resources might not be
      stored in the location specified in its name.
  """

    class DataplexTransferStatusValueValuesEnum(_messages.Enum):
        """Optional. Transfer status of the TagTemplate

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
    This map is an exhaustive list of the allowed fields. The map must contain
    at least one field and at most 500 fields. The keys to this map are tag
    template field IDs. The IDs have the following limitations: * Can contain
    uppercase and lowercase letters, numbers (0-9) and underscores (_). * Must
    be at least 1 character and at most 64 characters long. * Must start with
    a letter or underscore.

    Messages:
      AdditionalProperty: An additional property for a FieldsValue object.

    Fields:
      additionalProperties: Additional properties of type FieldsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FieldsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDatacatalogV1TagTemplateField attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDatacatalogV1TagTemplateField', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    dataplexTransferStatus = _messages.EnumField('DataplexTransferStatusValueValuesEnum', 1)
    displayName = _messages.StringField(2)
    fields = _messages.MessageField('FieldsValue', 3)
    isPubliclyReadable = _messages.BooleanField(4)
    name = _messages.StringField(5)