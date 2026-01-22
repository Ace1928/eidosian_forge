from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EntryType(_messages.Message):
    """Entry Type is a template for creating Entries.

  Messages:
    LabelsValue: Optional. User-defined labels for the EntryType.

  Fields:
    authorization: Immutable. Authorization defined for this type.
    createTime: Output only. The time when the EntryType was created.
    description: Optional. Description of the EntryType.
    displayName: Optional. User friendly display name.
    etag: Optional. This checksum is computed by the server based on the value
      of other fields, and may be sent on update and delete requests to ensure
      the client has an up-to-date value before proceeding.
    labels: Optional. User-defined labels for the EntryType.
    name: Output only. The relative resource name of the EntryType, of the
      form: projects/{project_number}/locations/{location_id}/entryTypes/{entr
      y_type_id}.
    platform: Optional. The platform that Entries of this type belongs to.
    requiredAspects: AspectInfo for the entry type.
    system: Optional. The system that Entries of this type belongs to.
      Examples include CloudSQL, MariaDB etc
    typeAliases: Optional. Indicates the class this Entry Type belongs to, for
      example, TABLE, DATABASE, MODEL.
    uid: Output only. System generated globally unique ID for the EntryType.
      This ID will be different if the EntryType is deleted and re-created
      with the same name.
    updateTime: Output only. The time when the EntryType was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User-defined labels for the EntryType.

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
    authorization = _messages.MessageField('GoogleCloudDataplexV1EntryTypeAuthorization', 1)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    etag = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    platform = _messages.StringField(8)
    requiredAspects = _messages.MessageField('GoogleCloudDataplexV1EntryTypeAspectInfo', 9, repeated=True)
    system = _messages.StringField(10)
    typeAliases = _messages.StringField(11, repeated=True)
    uid = _messages.StringField(12)
    updateTime = _messages.StringField(13)