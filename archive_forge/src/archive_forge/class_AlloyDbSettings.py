from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloyDbSettings(_messages.Message):
    """Settings for creating an AlloyDB cluster.

  Enums:
    DatabaseVersionValueValuesEnum: Optional. The database engine major
      version. This is an optional field. If a database version is not
      supplied at cluster creation time, then a default database version will
      be used.

  Messages:
    LabelsValue: Labels for the AlloyDB cluster created by DMS. An object
      containing a list of 'key', 'value' pairs.

  Fields:
    databaseVersion: Optional. The database engine major version. This is an
      optional field. If a database version is not supplied at cluster
      creation time, then a default database version will be used.
    encryptionConfig: Optional. The encryption config can be specified to
      encrypt the data disks and other persistent data resources of a cluster
      with a customer-managed encryption key (CMEK). When this field is not
      specified, the cluster will then use default encryption scheme to
      protect the user data.
    initialUser: Required. Input only. Initial user to setup during cluster
      creation. Required.
    labels: Labels for the AlloyDB cluster created by DMS. An object
      containing a list of 'key', 'value' pairs.
    primaryInstanceSettings: A PrimaryInstanceSettings attribute.
    vpcNetwork: Required. The resource link for the VPC network in which
      cluster resources are created and from which they are accessible via
      Private IP. The network must belong to the same project as the cluster.
      It is specified in the form:
      "projects/{project_number}/global/networks/{network_id}". This is
      required to create a cluster.
  """

    class DatabaseVersionValueValuesEnum(_messages.Enum):
        """Optional. The database engine major version. This is an optional
    field. If a database version is not supplied at cluster creation time,
    then a default database version will be used.

    Values:
      DATABASE_VERSION_UNSPECIFIED: This is an unknown database version.
      POSTGRES_14: The database version is Postgres 14.
      POSTGRES_15: The database version is Postgres 15.
    """
        DATABASE_VERSION_UNSPECIFIED = 0
        POSTGRES_14 = 1
        POSTGRES_15 = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels for the AlloyDB cluster created by DMS. An object containing a
    list of 'key', 'value' pairs.

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
    databaseVersion = _messages.EnumField('DatabaseVersionValueValuesEnum', 1)
    encryptionConfig = _messages.MessageField('EncryptionConfig', 2)
    initialUser = _messages.MessageField('UserPassword', 3)
    labels = _messages.MessageField('LabelsValue', 4)
    primaryInstanceSettings = _messages.MessageField('PrimaryInstanceSettings', 5)
    vpcNetwork = _messages.StringField(6)