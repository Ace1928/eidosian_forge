from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _HandleBackupId(self, args):
    """Restores a backup using v1beta4. The backup is specified with backup_id.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
          with.

    Returns:
      A dict object representing the operations resource describing the
      restoreBackup operation if the restoreBackup was successful.
    """
    client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    instance_ref = client.resource_parser.Parse(args.instance, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
    if not args.backup_instance:
        args.backup_instance = args.instance
    result_operation = sql_client.instances.RestoreBackup(sql_messages.SqlInstancesRestoreBackupRequest(project=instance_ref.project, instance=instance_ref.instance, instancesRestoreBackupRequest=sql_messages.InstancesRestoreBackupRequest(restoreBackupContext=sql_messages.RestoreBackupContext(backupRunId=args.backup_id, instanceId=args.backup_instance))))
    operation_ref = client.resource_parser.Create('sql.operations', operation=result_operation.name, project=instance_ref.project)
    if args.async_:
        return sql_client.operations.Get(sql_messages.SqlOperationsGetRequest(project=operation_ref.project, operation=operation_ref.operation))
    operations.OperationsV1Beta4.WaitForOperation(sql_client, operation_ref, 'Restoring Cloud SQL instance')
    log.status.write('Restored [{instance}].\n'.format(instance=instance_ref))
    return None