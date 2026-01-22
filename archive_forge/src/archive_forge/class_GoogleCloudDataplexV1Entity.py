from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Entity(_messages.Message):
    """Represents tables and fileset metadata contained within a zone.

  Enums:
    SystemValueValuesEnum: Required. Immutable. Identifies the storage system
      of the entity data.
    TypeValueValuesEnum: Required. Immutable. The type of entity.

  Fields:
    access: Output only. Identifies the access mechanism to the entity. Not
      user settable.
    asset: Required. Immutable. The ID of the asset associated with the
      storage location containing the entity data. The entity must be with in
      the same zone with the asset.
    catalogEntry: Output only. The name of the associated Data Catalog entry.
    compatibility: Output only. Metadata stores that the entity is compatible
      with.
    createTime: Output only. The time when the entity was created.
    dataPath: Required. Immutable. The storage path of the entity data. For
      Cloud Storage data, this is the fully-qualified path to the entity, such
      as gs://bucket/path/to/data. For BigQuery data, this is the name of the
      table resource, such as
      projects/project_id/datasets/dataset_id/tables/table_id.
    dataPathPattern: Optional. The set of items within the data path
      constituting the data in the entity, represented as a glob path.
      Example: gs://bucket/path/to/data/**/*.csv.
    description: Optional. User friendly longer description text. Must be
      shorter than or equal to 1024 characters.
    displayName: Optional. Display name must be shorter than or equal to 256
      characters.
    etag: Optional. The etag associated with the entity, which can be
      retrieved with a GetEntity request. Required for update and delete
      requests.
    format: Required. Identifies the storage format of the entity data. It
      does not apply to entities with data stored in BigQuery.
    id: Required. A user-provided entity ID. It is mutable, and will be used
      as the published table name. Specifying a new ID in an update entity
      request will override the existing value. The ID must contain only
      letters (a-z, A-Z), numbers (0-9), and underscores, and consist of 256
      or fewer characters.
    name: Output only. The resource name of the entity, of the form: projects/
      {project_number}/locations/{location_id}/lakes/{lake_id}/zones/{zone_id}
      /entities/{id}.
    schema: Required. The description of the data structure and layout. The
      schema is not included in list responses. It is only included in SCHEMA
      and FULL entity views of a GetEntity response.
    system: Required. Immutable. Identifies the storage system of the entity
      data.
    type: Required. Immutable. The type of entity.
    uid: Output only. System generated unique ID for the Entity. This ID will
      be different if the Entity is deleted and re-created with the same name.
    updateTime: Output only. The time when the entity was last updated.
  """

    class SystemValueValuesEnum(_messages.Enum):
        """Required. Immutable. Identifies the storage system of the entity data.

    Values:
      STORAGE_SYSTEM_UNSPECIFIED: Storage system unspecified.
      CLOUD_STORAGE: The entity data is contained within a Cloud Storage
        bucket.
      BIGQUERY: The entity data is contained within a BigQuery dataset.
    """
        STORAGE_SYSTEM_UNSPECIFIED = 0
        CLOUD_STORAGE = 1
        BIGQUERY = 2

    class TypeValueValuesEnum(_messages.Enum):
        """Required. Immutable. The type of entity.

    Values:
      TYPE_UNSPECIFIED: Type unspecified.
      TABLE: Structured and semi-structured data.
      FILESET: Unstructured data.
    """
        TYPE_UNSPECIFIED = 0
        TABLE = 1
        FILESET = 2
    access = _messages.MessageField('GoogleCloudDataplexV1StorageAccess', 1)
    asset = _messages.StringField(2)
    catalogEntry = _messages.StringField(3)
    compatibility = _messages.MessageField('GoogleCloudDataplexV1EntityCompatibilityStatus', 4)
    createTime = _messages.StringField(5)
    dataPath = _messages.StringField(6)
    dataPathPattern = _messages.StringField(7)
    description = _messages.StringField(8)
    displayName = _messages.StringField(9)
    etag = _messages.StringField(10)
    format = _messages.MessageField('GoogleCloudDataplexV1StorageFormat', 11)
    id = _messages.StringField(12)
    name = _messages.StringField(13)
    schema = _messages.MessageField('GoogleCloudDataplexV1Schema', 14)
    system = _messages.EnumField('SystemValueValuesEnum', 15)
    type = _messages.EnumField('TypeValueValuesEnum', 16)
    uid = _messages.StringField(17)
    updateTime = _messages.StringField(18)