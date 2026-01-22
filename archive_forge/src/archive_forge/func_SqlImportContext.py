from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def SqlImportContext(sql_messages, uri, database=None, user=None, parallel=False, threads=None):
    """Generates the ImportContext for the given args, for importing from SQL.

  Args:
    sql_messages: module, The messages module that should be used.
    uri: The URI of the bucket to import from; the output of the 'uri' arg.
    database: The database to import to; the output of the '--database' flag.
    user: The Postgres user to import as; the output of the '--user' flag.
    parallel: Whether to use parallel import or not; the output of the
      '--parallel' flag.
    threads: The number of threads to use; the output of the '--threads' flag.
      Only applicable for parallel import.

  Returns:
    ImportContext, for use in InstancesImportRequest.importContext.
  """
    if parallel:
        return sql_messages.ImportContext(kind='sql#importContext', uri=uri, database=database, fileType=sql_messages.ImportContext.FileTypeValueValuesEnum.SQL, importUser=user, sqlImportOptions=sql_messages.ImportContext.SqlImportOptionsValue(parallel=parallel, threads=threads))
    else:
        return sql_messages.ImportContext(kind='sql#importContext', uri=uri, database=database, fileType=sql_messages.ImportContext.FileTypeValueValuesEnum.SQL, importUser=user, sqlImportOptions=sql_messages.ImportContext.SqlImportOptionsValue(threads=threads))