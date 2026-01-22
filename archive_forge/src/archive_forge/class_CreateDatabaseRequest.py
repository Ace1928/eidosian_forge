from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateDatabaseRequest(_messages.Message):
    """The request for CreateDatabase.

  Enums:
    DatabaseDialectValueValuesEnum: Optional. The dialect of the Cloud Spanner
      Database.

  Fields:
    createStatement: Required. A `CREATE DATABASE` statement, which specifies
      the ID of the new database. The database ID must conform to the regular
      expression `a-z*[a-z0-9]` and be between 2 and 30 characters in length.
      If the database ID is a reserved word or if it contains a hyphen, the
      database ID must be enclosed in backticks (`` ` ``).
    databaseDialect: Optional. The dialect of the Cloud Spanner Database.
    encryptionConfig: Optional. The encryption configuration for the database.
      If this field is not specified, Cloud Spanner will encrypt/decrypt all
      data at rest using Google default encryption.
    extraStatements: Optional. A list of DDL statements to run inside the
      newly created database. Statements can create tables, indexes, etc.
      These statements execute atomically with the creation of the database:
      if there is an error in any statement, the database is not created.
    protoDescriptors: Optional. Proto descriptors used by CREATE/ALTER PROTO
      BUNDLE statements in 'extra_statements' above. Contains a protobuf-
      serialized [google.protobuf.FileDescriptorSet](https://github.com/protoc
      olbuffers/protobuf/blob/main/src/google/protobuf/descriptor.proto). To
      generate it, [install](https://grpc.io/docs/protoc-installation/) and
      run `protoc` with --include_imports and --descriptor_set_out. For
      example, to generate for moon/shot/app.proto, run ``` $protoc
      --proto_path=/app_path --proto_path=/lib_path \\ --include_imports \\
      --descriptor_set_out=descriptors.data \\ moon/shot/app.proto ``` For more
      details, see protobuffer [self
      description](https://developers.google.com/protocol-
      buffers/docs/techniques#self-description).
  """

    class DatabaseDialectValueValuesEnum(_messages.Enum):
        """Optional. The dialect of the Cloud Spanner Database.

    Values:
      DATABASE_DIALECT_UNSPECIFIED: Default value. This value will create a
        database with the GOOGLE_STANDARD_SQL dialect.
      GOOGLE_STANDARD_SQL: GoogleSQL supported SQL.
      POSTGRESQL: PostgreSQL supported SQL.
    """
        DATABASE_DIALECT_UNSPECIFIED = 0
        GOOGLE_STANDARD_SQL = 1
        POSTGRESQL = 2
    createStatement = _messages.StringField(1)
    databaseDialect = _messages.EnumField('DatabaseDialectValueValuesEnum', 2)
    encryptionConfig = _messages.MessageField('EncryptionConfig', 3)
    extraStatements = _messages.StringField(4, repeated=True)
    protoDescriptors = _messages.BytesField(5)