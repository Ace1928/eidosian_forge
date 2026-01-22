from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalCatalogTableOptions(_messages.Message):
    """Metadata about open source compatible table. The fields contained in
  these options correspond to hive metastore's table level properties.

  Messages:
    ParametersValue: Optional. A map of key value pairs defining the
      parameters and properties of the open source table. Corresponds with
      hive meta store table parameters. Maximum size of 4Mib.

  Fields:
    connectionId: Optional. The connection specifying the credentials to be
      used to read external storage, such as Azure Blob, Cloud Storage, or S3.
      The connection is needed to read the open source table from BigQuery
      Engine. The connection_id can have the form `..` or
      `projects//locations//connections/`.
    parameters: Optional. A map of key value pairs defining the parameters and
      properties of the open source table. Corresponds with hive meta store
      table parameters. Maximum size of 4Mib.
    storageDescriptor: Optional. A storage descriptor containing information
      about the physical storage of this table.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """Optional. A map of key value pairs defining the parameters and
    properties of the open source table. Corresponds with hive meta store
    table parameters. Maximum size of 4Mib.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Additional properties of type ParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    connectionId = _messages.StringField(1)
    parameters = _messages.MessageField('ParametersValue', 2)
    storageDescriptor = _messages.MessageField('StorageDescriptor', 3)