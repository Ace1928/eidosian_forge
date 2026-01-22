from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import log
class RestoreTables(base.RestoreCommand):
    """Restore a Cloud Bigtable backup to a new table."""
    detailed_help = {'DESCRIPTION': textwrap.dedent('\n          This command restores a Cloud Bigtable backup to a new table.\n          '), 'EXAMPLES': textwrap.dedent("\n          To restore table 'table2' from backup 'backup1', run:\n\n            $ {command} --source-instance=instance1 --source-cluster=cluster1 --source=backup1 --destination-instance=instance1 --destination=table2\n\n          To restore table 'table2' from backup 'backup1' in a different project, run:\n\n            $ {command} --source=projects/project1/instances/instance1/clusters/cluster1/backups/backup1 --destination=projects/project2/instances/instance2/tables/table2\n          ")}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        arguments.AddTableRestoreResourceArg(parser)
        arguments.ArgAdder(parser).AddAsync()

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        cli = util.GetAdminClient()
        msgs = util.GetAdminMessages()
        backup_ref = args.CONCEPTS.source.Parse()
        table_ref = args.CONCEPTS.destination.Parse()
        restore_request = msgs.RestoreTableRequest(backup=backup_ref.RelativeName(), tableId=table_ref.Name())
        msg = msgs.BigtableadminProjectsInstancesTablesRestoreRequest(parent=table_ref.Parent().RelativeName(), restoreTableRequest=restore_request)
        operation = cli.projects_instances_tables.Restore(msg)
        operation_ref = util.GetOperationRef(operation)
        if args.async_:
            log.CreatedResource(operation_ref.RelativeName(), kind='bigtable table {0}'.format(table_ref.Name()), is_async=True)
            return
        return util.AwaitTable(operation_ref, 'Creating bigtable table {0}'.format(table_ref.Name()))