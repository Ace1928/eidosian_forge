from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesBackupOperationsListRequest(_messages.Message):
    """A SpannerProjectsInstancesBackupOperationsListRequest object.

  Fields:
    filter: An expression that filters the list of returned backup operations.
      A filter expression consists of a field name, a comparison operator, and
      a value for filtering. The value must be a string, a number, or a
      boolean. The comparison operator must be one of: `<`, `>`, `<=`, `>=`,
      `!=`, `=`, or `:`. Colon `:` is the contains operator. Filter rules are
      not case sensitive. The following fields in the operation are eligible
      for filtering: * `name` - The name of the long-running operation *
      `done` - False if the operation is in progress, else true. *
      `metadata.@type` - the type of metadata. For example, the type string
      for CreateBackupMetadata is `type.googleapis.com/google.spanner.admin.da
      tabase.v1.CreateBackupMetadata`. * `metadata.` - any field in
      metadata.value. `metadata.@type` must be specified first if filtering on
      metadata fields. * `error` - Error associated with the long-running
      operation. * `response.@type` - the type of response. * `response.` -
      any field in response.value. You can combine multiple expressions by
      enclosing each expression in parentheses. By default, expressions are
      combined with AND logic, but you can specify AND, OR, and NOT logic
      explicitly. Here are a few examples: * `done:true` - The operation is
      complete. * `(metadata.@type=type.googleapis.com/google.spanner.admin.da
      tabase.v1.CreateBackupMetadata) AND` \\ `metadata.database:prod` -
      Returns operations where: * The operation's metadata type is
      CreateBackupMetadata. * The source database name of backup contains the
      string "prod". * `(metadata.@type=type.googleapis.com/google.spanner.adm
      in.database.v1.CreateBackupMetadata) AND` \\ `(metadata.name:howl) AND` \\
      `(metadata.progress.start_time < \\"2018-03-28T14:50:00Z\\") AND` \\
      `(error:*)` - Returns operations where: * The operation's metadata type
      is CreateBackupMetadata. * The backup name contains the string "howl". *
      The operation started before 2018-03-28T14:50:00Z. * The operation
      resulted in an error. * `(metadata.@type=type.googleapis.com/google.span
      ner.admin.database.v1.CopyBackupMetadata) AND` \\
      `(metadata.source_backup:test) AND` \\ `(metadata.progress.start_time <
      \\"2022-01-18T14:50:00Z\\") AND` \\ `(error:*)` - Returns operations where:
      * The operation's metadata type is CopyBackupMetadata. * The source
      backup name contains the string "test". * The operation started before
      2022-01-18T14:50:00Z. * The operation resulted in an error. * `((metadat
      a.@type=type.googleapis.com/google.spanner.admin.database.v1.CreateBacku
      pMetadata) AND` \\ `(metadata.database:test_db)) OR` \\ `((metadata.@type=
      type.googleapis.com/google.spanner.admin.database.v1.CopyBackupMetadata)
      AND` \\ `(metadata.source_backup:test_bkp)) AND` \\ `(error:*)` - Returns
      operations where: * The operation's metadata matches either of criteria:
      * The operation's metadata type is CreateBackupMetadata AND the source
      database name of the backup contains the string "test_db" * The
      operation's metadata type is CopyBackupMetadata AND the source backup
      name contains the string "test_bkp" * The operation resulted in an
      error.
    pageSize: Number of operations to be returned in the response. If 0 or
      less, defaults to the server's maximum allowed page size.
    pageToken: If non-empty, `page_token` should contain a next_page_token
      from a previous ListBackupOperationsResponse to the same `parent` and
      with the same `filter`.
    parent: Required. The instance of the backup operations. Values are of the
      form `projects//instances/`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)