from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def BakImportContext(sql_messages, uri, database, cert_path, pvk_path, pvk_password, striped, no_recovery, recovery_only, bak_type, stop_at, stop_at_mark):
    """Generates the ImportContext for the given args, for importing from BAK.

  Args:
    sql_messages: module, The messages module that should be used.
    uri: The URI of the bucket to import from; the output of the `uri` arg.
    database: The database to import to; the output of the `--database` flag.
    cert_path: The certificate used for encrypted .bak; the output of the
      `--cert-path` flag.
    pvk_path: The private key used for encrypted .bak; the output of the
      `--pvk-path` flag.
    pvk_password: The private key password used for encrypted .bak; the output
      of the `--pvk-password` or `--prompt-for-pvk-password` flag.
    striped: Whether or not the import is striped.
    no_recovery: Whether the import executes with NORECOVERY keyword.
    recovery_only: Whether the import skip download and bring database online.
    bak_type: Type of the bak file.
    stop_at: Equivalent to SQL Server STOPAT keyword.
    stop_at_mark: Equivalent to SQL Server STOPATMARK keyword.

  Returns:
    ImportContext, for use in InstancesImportRequest.importContext.
  """
    bak_import_options = None
    if cert_path and pvk_path and pvk_password:
        bak_import_options = sql_messages.ImportContext.BakImportOptionsValue(encryptionOptions=sql_messages.ImportContext.BakImportOptionsValue.EncryptionOptionsValue(certPath=cert_path, pvkPath=pvk_path, pvkPassword=pvk_password))
    else:
        bak_import_options = sql_messages.ImportContext.BakImportOptionsValue()
    if striped:
        bak_import_options.striped = striped
    bak_import_options.noRecovery = no_recovery
    bak_import_options.recoveryOnly = recovery_only
    bak_import_options.bakType = ParseBakType(sql_messages, bak_type)
    if stop_at is not None:
        bak_import_options.stopAt = stop_at.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    bak_import_options.stopAtMark = stop_at_mark
    return sql_messages.ImportContext(kind='sql#importContext', uri=uri, database=database, fileType=sql_messages.ImportContext.FileTypeValueValuesEnum.BAK, bakImportOptions=bak_import_options)