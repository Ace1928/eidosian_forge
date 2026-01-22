from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Index(_messages.Message):
    """A representation of a collection of database items organized in a way
  that allows for approximate nearest neighbor (a.k.a ANN) algorithms search.

  Enums:
    IndexUpdateMethodValueValuesEnum: Immutable. The update method to use with
      this Index. If not set, BATCH_UPDATE will be used by default.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      Indexes. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. See https://goo.gl/xmQnxf for more information and examples of
      labels.

  Fields:
    createTime: Output only. Timestamp when this Index was created.
    deployedIndexes: Output only. The pointers to DeployedIndexes created from
      this Index. An Index can be only deleted if all its DeployedIndexes had
      been undeployed first.
    description: The description of the Index.
    displayName: Required. The display name of the Index. The name can be up
      to 128 characters long and can consist of any UTF-8 characters.
    encryptionSpec: Immutable. Customer-managed encryption key spec for an
      Index. If set, this Index and all sub-resources of this Index will be
      secured by this key.
    etag: Used to perform consistent read-modify-write updates. If not set, a
      blind "overwrite" update happens.
    indexStats: Output only. Stats of the index resource.
    indexUpdateMethod: Immutable. The update method to use with this Index. If
      not set, BATCH_UPDATE will be used by default.
    labels: The labels with user-defined metadata to organize your Indexes.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. See
      https://goo.gl/xmQnxf for more information and examples of labels.
    metadata: An additional information about the Index; the schema of the
      metadata can be found in metadata_schema.
    metadataSchemaUri: Immutable. Points to a YAML file stored on Google Cloud
      Storage describing additional information about the Index, that is
      specific to it. Unset if the Index does not have any additional
      information. The schema is defined as an OpenAPI 3.0.2 [Schema
      Object](https://github.com/OAI/OpenAPI-
      Specification/blob/main/versions/3.0.2.md#schemaObject). Note: The URI
      given on output will be immutable and probably different, including the
      URI scheme, than the one given on input. The output URI will point to a
      location where the user only has a read access.
    name: Output only. The resource name of the Index.
    updateTime: Output only. Timestamp when this Index was most recently
      updated. This also includes any update to the contents of the Index.
      Note that Operations working on this Index may have their
      Operations.metadata.generic_metadata.update_time a little after the
      value of this timestamp, yet that does not mean their results are not
      already reflected in the Index. Result of any successfully completed
      Operation on the Index is reflected in it.
  """

    class IndexUpdateMethodValueValuesEnum(_messages.Enum):
        """Immutable. The update method to use with this Index. If not set,
    BATCH_UPDATE will be used by default.

    Values:
      INDEX_UPDATE_METHOD_UNSPECIFIED: Should not be used.
      BATCH_UPDATE: BatchUpdate: user can call UpdateIndex with files on Cloud
        Storage of Datapoints to update.
      STREAM_UPDATE: StreamUpdate: user can call
        UpsertDatapoints/DeleteDatapoints to update the Index and the updates
        will be applied in corresponding DeployedIndexes in nearly real-time.
    """
        INDEX_UPDATE_METHOD_UNSPECIFIED = 0
        BATCH_UPDATE = 1
        STREAM_UPDATE = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your Indexes. Label
    keys and values can be no longer than 64 characters (Unicode codepoints),
    can only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. See https://goo.gl/xmQnxf
    for more information and examples of labels.

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
    deployedIndexes = _messages.MessageField('GoogleCloudAiplatformV1DeployedIndexRef', 2, repeated=True)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    encryptionSpec = _messages.MessageField('GoogleCloudAiplatformV1EncryptionSpec', 5)
    etag = _messages.StringField(6)
    indexStats = _messages.MessageField('GoogleCloudAiplatformV1IndexStats', 7)
    indexUpdateMethod = _messages.EnumField('IndexUpdateMethodValueValuesEnum', 8)
    labels = _messages.MessageField('LabelsValue', 9)
    metadata = _messages.MessageField('extra_types.JsonValue', 10)
    metadataSchemaUri = _messages.StringField(11)
    name = _messages.StringField(12)
    updateTime = _messages.StringField(13)