from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import instances as command_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class RestoreBackup(base.RestoreCommand):
    """Restores a backup of a Cloud SQL instance."""

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go
          on the command line after this command. Positional arguments are
          allowed.
    """
        flags.AddBackupId(parser, help_text='The ID of the backup run to restore from.')
        parser.add_argument('--restore-instance', required=True, completer=flags.InstanceCompleter, help='The ID of the target Cloud SQL instance that the backup is restored to.')
        parser.add_argument('--backup-instance', completer=flags.InstanceCompleter, help='The ID of the instance that the backup was taken from. This argument must be specified when the backup instance is different from the restore instance. If it is not specified, the backup instance is considered the same as the restore instance.')
        flags.AddProjectLevelBackupEndpoint(parser)
        base.ASYNC_FLAG.AddToParser(parser)
        AddInstanceSettingsArgs(parser)

    def Run(self, args):
        """Restores a backup of a Cloud SQL instance.

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
        validate.ValidateInstanceName(args.restore_instance)
        instance_ref = client.resource_parser.Parse(args.restore_instance, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
        if not console_io.PromptContinue('All current data on the instance will be lost when the backup is restored.'):
            return None
        if not args.backup_instance:
            args.backup_instance = args.restore_instance
        if args.project_level:
            instance_resource = command_util.InstancesV1Beta4.ConstructCreateInstanceFromArgs(sql_messages, args, instance_ref=instance_ref)
            result_operation = sql_client.instances.RestoreBackup(sql_messages.SqlInstancesRestoreBackupRequest(project=instance_ref.project, instance=instance_ref.instance, instancesRestoreBackupRequest=sql_messages.InstancesRestoreBackupRequest(backup=args.id, restoreInstanceSettings=instance_resource)))
        else:
            backup_run_id = int(args.id)
            result_operation = sql_client.instances.RestoreBackup(sql_messages.SqlInstancesRestoreBackupRequest(project=instance_ref.project, instance=instance_ref.instance, instancesRestoreBackupRequest=sql_messages.InstancesRestoreBackupRequest(restoreBackupContext=sql_messages.RestoreBackupContext(backupRunId=backup_run_id, instanceId=args.backup_instance))))
        operation_ref = client.resource_parser.Create('sql.operations', operation=result_operation.name, project=instance_ref.project)
        if args.async_:
            return sql_client.operations.Get(sql_messages.SqlOperationsGetRequest(project=operation_ref.project, operation=operation_ref.operation))
        operations.OperationsV1Beta4.WaitForOperation(sql_client, operation_ref, 'Restoring Cloud SQL instance')
        log.status.write('Restored [{instance}].\n'.format(instance=instance_ref))
        return None