from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import export_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def RunExportCommand(args, client, export_context):
    """Exports data from a Cloud SQL instance.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.
    client: SqlClient instance, with sql_client and sql_messages props, for use
      in generating messages and making API calls.
    export_context: ExportContext; format-specific export metadata.

  Returns:
    A dict representing the export operation resource, if '--async' is used,
    or else None.

  Raises:
    HttpException: An HTTP error response was received while executing API
        request.
    ToolException: An error other than HTTP error occurred while executing the
        command.
  """
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    validate.ValidateInstanceName(args.instance)
    instance_ref = client.resource_parser.Parse(args.instance, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
    export_request = sql_messages.SqlInstancesExportRequest(instance=instance_ref.instance, project=instance_ref.project, instancesExportRequest=sql_messages.InstancesExportRequest(exportContext=export_context))
    result_operation = sql_client.instances.Export(export_request)
    operation_ref = client.resource_parser.Create('sql.operations', operation=result_operation.name, project=instance_ref.project)
    if args.async_:
        return sql_client.operations.Get(sql_messages.SqlOperationsGetRequest(project=operation_ref.project, operation=operation_ref.operation))
    operations.OperationsV1Beta4.WaitForOperation(sql_client, operation_ref, 'Exporting Cloud SQL instance')
    log.status.write('Exported [{instance}] to [{bucket}].\n'.format(instance=instance_ref, bucket=args.uri))
    return None