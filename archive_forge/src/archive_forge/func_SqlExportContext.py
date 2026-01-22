from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def SqlExportContext(sql_messages, uri, database=None, table=None, offload=False, parallel=False, threads=None):
    """Generates the ExportContext for the given args, for exporting to SQL.

  Args:
    sql_messages: module, The messages module that should be used.
    uri: The URI of the bucket to export to; the output of the 'uri' arg.
    database: The list of databases to export from; the output of the
      '--database' flag.
    table: The list of tables to export from; the output of the '--table' flag.
    offload: bool, The export offload flag.
    parallel: Whether to use parallel export or not.
    threads: The number of threads to use. Only applicable for parallel export.

  Returns:
    ExportContext, for use in InstancesExportRequest.exportContext.
  """
    if parallel:
        return sql_messages.ExportContext(kind='sql#exportContext', uri=uri, databases=database or [], offload=offload, fileType=sql_messages.ExportContext.FileTypeValueValuesEnum.SQL, sqlExportOptions=sql_messages.ExportContext.SqlExportOptionsValue(tables=table or [], parallel=parallel, threads=threads))
    else:
        return sql_messages.ExportContext(kind='sql#exportContext', uri=uri, databases=database or [], offload=offload, fileType=sql_messages.ExportContext.FileTypeValueValuesEnum.SQL, sqlExportOptions=sql_messages.ExportContext.SqlExportOptionsValue(tables=table or [], threads=threads))