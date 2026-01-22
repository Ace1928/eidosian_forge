from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Context(_messages.Message):
    """Instance of a general context.

  Messages:
    LabelsValue: The labels with user-defined metadata to organize your
      Contexts. Label keys and values can be no longer than 64 characters
      (Unicode codepoints), can only contain lowercase letters, numeric
      characters, underscores and dashes. International characters are
      allowed. No more than 64 user labels can be associated with one Context
      (System labels are excluded).
    MetadataValue: Properties of the Context. Top level metadata keys' heading
      and trailing spaces will be trimmed. The size of this field should not
      exceed 200KB.

  Fields:
    createTime: Output only. Timestamp when this Context was created.
    description: Description of the Context
    displayName: User provided display name of the Context. May be up to 128
      Unicode characters.
    etag: An eTag used to perform consistent read-modify-write updates. If not
      set, a blind "overwrite" update happens.
    labels: The labels with user-defined metadata to organize your Contexts.
      Label keys and values can be no longer than 64 characters (Unicode
      codepoints), can only contain lowercase letters, numeric characters,
      underscores and dashes. International characters are allowed. No more
      than 64 user labels can be associated with one Context (System labels
      are excluded).
    metadata: Properties of the Context. Top level metadata keys' heading and
      trailing spaces will be trimmed. The size of this field should not
      exceed 200KB.
    name: Immutable. The resource name of the Context.
    parentContexts: Output only. A list of resource names of Contexts that are
      parents of this Context. A Context may have at most 10 parent_contexts.
    schemaTitle: The title of the schema describing the metadata. Schema title
      and version is expected to be registered in earlier Create Schema calls.
      And both are used together as unique identifiers to identify schemas
      within the local metadata store.
    schemaVersion: The version of the schema in schema_name to use. Schema
      title and version is expected to be registered in earlier Create Schema
      calls. And both are used together as unique identifiers to identify
      schemas within the local metadata store.
    updateTime: Output only. Timestamp when this Context was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels with user-defined metadata to organize your Contexts. Label
    keys and values can be no longer than 64 characters (Unicode codepoints),
    can only contain lowercase letters, numeric characters, underscores and
    dashes. International characters are allowed. No more than 64 user labels
    can be associated with one Context (System labels are excluded).

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadataValue(_messages.Message):
        """Properties of the Context. Top level metadata keys' heading and
    trailing spaces will be trimmed. The size of this field should not exceed
    200KB.

    Messages:
      AdditionalProperty: An additional property for a MetadataValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    metadata = _messages.MessageField('MetadataValue', 6)
    name = _messages.StringField(7)
    parentContexts = _messages.StringField(8, repeated=True)
    schemaTitle = _messages.StringField(9)
    schemaVersion = _messages.StringField(10)
    updateTime = _messages.StringField(11)