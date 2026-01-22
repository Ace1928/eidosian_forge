from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import instances
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def RunBaseSwitchoverCommand(args):
    """Switches over a Cloud SQL instance to one of its replicas.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    A dict object representing the operations resource describing the
    switchover operation if the switchover was successful.
  """
    client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    validate.ValidateInstanceName(args.replica)
    instance_ref = client.resource_parser.Parse(args.replica, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
    instance_resource = sql_client.instances.Get(sql_messages.SqlInstancesGetRequest(project=instance_ref.project, instance=instance_ref.instance))
    if not instances.InstancesV1Beta4.IsSqlServerDatabaseVersion(instance_resource.databaseVersion) and (not instances.InstancesV1Beta4.IsMysqlDatabaseVersion(instance_resource.databaseVersion)):
        raise exceptions.OperationError('Switchover operation is currently supported for Cloud SQL for SQL Server and MySQL instances only')
    sys.stderr.write(textwrap.TextWrapper().fill('Switching over to a replica leads to a short period of downtime and results in the primary and replica instances "switching" roles. Before switching over to the replica, you must verify that both the primary and replica instances are online. Otherwise, use a promote operation.') + '\n\n')
    console_io.PromptContinue(message='', default=True, cancel_on_no=True)
    db_timeout_str = args.db_timeout
    if db_timeout_str is not None:
        db_timeout_str = str(args.db_timeout) + 's'
    result = sql_client.instances.Switchover(sql_messages.SqlInstancesSwitchoverRequest(project=instance_ref.project, instance=instance_ref.instance, dbTimeout=db_timeout_str))
    operation_ref = client.resource_parser.Create('sql.operations', operation=result.name, project=instance_ref.project)
    if args.async_:
        return sql_client.operations.Get(sql_messages.SqlOperationsGetRequest(project=operation_ref.project, operation=operation_ref.operation))
    operations.OperationsV1Beta4.WaitForOperation(sql_client, operation_ref, 'Switching over to Cloud SQL replica')
    log.status.write('Switched over [{instance}].\n'.format(instance=instance_ref))