from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EntryGroup(_messages.Message):
    """An Entry Group represents a logical grouping of one or more Entries.

  Enums:
    TransferStatusValueValuesEnum: Output only. Denotes the transfer status of
      the Entry Group. It is unspecified for Entry Group created from Dataplex
      API.

  Messages:
    LabelsValue: Optional. User-defined labels for the EntryGroup.

  Fields:
    createTime: Output only. The time when the EntryGroup was created.
    description: Optional. Description of the EntryGroup.
    displayName: Optional. User friendly display name.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding.
    labels: Optional. User-defined labels for the EntryGroup.
    name: Output only. The relative resource name of the EntryGroup, of the
      form: projects/{project_number}/locations/{location_id}/entryGroups/{ent
      ry_group_id}.
    transferStatus: Output only. Denotes the transfer status of the Entry
      Group. It is unspecified for Entry Group created from Dataplex API.
    uid: Output only. System generated globally unique ID for the EntryGroup.
      This ID will be different if the EntryGroup is deleted and re-created
      with the same name.
    updateTime: Output only. The time when the EntryGroup was last updated.
  """

    class TransferStatusValueValuesEnum(_messages.Enum):
        """Output only. Denotes the transfer status of the Entry Group. It is
    unspecified for Entry Group created from Dataplex API.

    Values:
      TRANSFER_STATUS_UNSPECIFIED: The default value. It is set for resources
        that were not subject for migration from Data Catalog service.
      TRANSFER_STATUS_MIGRATED: Indicates that a resource was migrated from
        Data Catalog service but it hasn't been transferred yet. In particular
        the resource cannot be updated from Dataplex API.
      TRANSFER_STATUS_TRANSFERRED: Indicates that a resource was transferred
        from Data Catalog service. The resource can only be updated from
        Dataplex API.
    """
        TRANSFER_STATUS_UNSPECIFIED = 0
        TRANSFER_STATUS_MIGRATED = 1
        TRANSFER_STATUS_TRANSFERRED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. User-defined labels for the EntryGroup.

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
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    name = _messages.StringField(6)
    transferStatus = _messages.EnumField('TransferStatusValueValuesEnum', 7)
    uid = _messages.StringField(8)
    updateTime = _messages.StringField(9)