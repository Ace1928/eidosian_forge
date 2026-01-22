from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Entry(_messages.Message):
    """An entry is a representation of a data asset which can be described by
  various metadata.

  Messages:
    AspectsValue: Optional. The Aspects attached to the Entry. The format for
      the key can be one of the following: 1.
      {projectId}.{locationId}.{aspectTypeId} (if the aspect is attached
      directly to the entry) 2. {projectId}.{locationId}.{aspectTypeId}@{path}
      (if the aspect is attached to an entry's path)

  Fields:
    aspects: Optional. The Aspects attached to the Entry. The format for the
      key can be one of the following: 1.
      {projectId}.{locationId}.{aspectTypeId} (if the aspect is attached
      directly to the entry) 2. {projectId}.{locationId}.{aspectTypeId}@{path}
      (if the aspect is attached to an entry's path)
    createTime: Output only. The time when the Entry was created.
    entrySource: Optional. Source system related information for an entry.
    entryType: Required. Immutable. The resource name of the EntryType used to
      create this Entry.
    fullyQualifiedName: Optional. A name for the entry that can reference it
      in an external system. The maximum size of the field is 4000 characters.
    name: Identifier. The relative resource name of the Entry, of the form: pr
      ojects/{project}/locations/{location}/entryGroups/{entry_group}/entries/
      {entry}.
    parentEntry: Optional. Immutable. The resource name of the parent entry.
    updateTime: Output only. The time when the Entry was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AspectsValue(_messages.Message):
        """Optional. The Aspects attached to the Entry. The format for the key
    can be one of the following: 1. {projectId}.{locationId}.{aspectTypeId}
    (if the aspect is attached directly to the entry) 2.
    {projectId}.{locationId}.{aspectTypeId}@{path} (if the aspect is attached
    to an entry's path)

    Messages:
      AdditionalProperty: An additional property for a AspectsValue object.

    Fields:
      additionalProperties: Additional properties of type AspectsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AspectsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDataplexV1Aspect attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDataplexV1Aspect', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    aspects = _messages.MessageField('AspectsValue', 1)
    createTime = _messages.StringField(2)
    entrySource = _messages.MessageField('GoogleCloudDataplexV1EntrySource', 3)
    entryType = _messages.StringField(4)
    fullyQualifiedName = _messages.StringField(5)
    name = _messages.StringField(6)
    parentEntry = _messages.StringField(7)
    updateTime = _messages.StringField(8)