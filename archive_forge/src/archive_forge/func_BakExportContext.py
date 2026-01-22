from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def BakExportContext(sql_messages, uri, database, stripe_count, striped, bak_type, differential_base):
    """Generates the ExportContext for the given args, for exporting to BAK.

  Args:
    sql_messages: module, The messages module that should be used.
    uri: The URI of the bucket to export to; the output of the 'uri' arg.
    database: The list of databases to export from; the output of the
      '--database' flag.
    stripe_count: How many stripes to perform the export with.
    striped: Whether the export should be striped.
    bak_type: Type of bak file that will be exported. SQL Server only.
    differential_base: Whether the bak file export can be used as differential
      base for future differential backup. SQL Server only.

  Returns:
    ExportContext, for use in InstancesExportRequest.exportContext.
  """
    bak_export_options = sql_messages.ExportContext.BakExportOptionsValue()
    if striped is not None:
        bak_export_options.striped = striped
    if stripe_count is not None:
        bak_export_options.stripeCount = stripe_count
    bak_export_options.differentialBase = differential_base
    bak_export_options.bakType = ParseBakType(sql_messages, bak_type)
    return sql_messages.ExportContext(kind='sql#exportContext', uri=uri, databases=database, fileType=sql_messages.ExportContext.FileTypeValueValuesEnum.BAK, bakExportOptions=bak_export_options)