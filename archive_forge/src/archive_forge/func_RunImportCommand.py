from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.sql import import_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def RunImportCommand(args, client, import_context):
    """Imports data into a Cloud SQL instance.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.
    client: SqlClient instance, with sql_client and sql_messages props, for use
      in generating messages and making API calls.
    import_context: ImportContext; format-specific import metadata.

  Returns:
    A dict representing the import operation resource, if '--async' is used,
    or else None.

  Raises:
    HttpException: An HTTP error response was received while executing API
        request.
    ToolException: An error other than HTTP error occurred while executing the
        command.
  """
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    is_bak_import = import_context.fileType == sql_messages.ImportContext.FileTypeValueValuesEnum.BAK
    validate.ValidateInstanceName(args.instance)
    if is_bak_import:
        validate.ValidateURI(args.uri, args.recovery_only)
    instance_ref = client.resource_parser.Parse(args.instance, params={'project': properties.VALUES.core.project.GetOrFail}, collection='sql.instances')
    if is_bak_import and args.recovery_only:
        console_io.PromptContinue(message='Bring database [{database}] online with recovery-only.'.format(database=args.database), default=True, cancel_on_no=True)
    else:
        console_io.PromptContinue(message='Data from [{uri}] will be imported to [{instance}].'.format(uri=args.uri, instance=args.instance), default=True, cancel_on_no=True)
    import_request = sql_messages.SqlInstancesImportRequest(instance=instance_ref.instance, project=instance_ref.project, instancesImportRequest=sql_messages.InstancesImportRequest(importContext=import_context))
    result_operation = sql_client.instances.Import(import_request)
    operation_ref = client.resource_parser.Create('sql.operations', operation=result_operation.name, project=instance_ref.project)
    if args.async_:
        return sql_client.operations.Get(sql_messages.SqlOperationsGetRequest(project=operation_ref.project, operation=operation_ref.operation))
    message = 'Importing data into Cloud SQL instance'
    if is_bak_import and args.recovery_only:
        message = 'Bring database online'
    operations.OperationsV1Beta4.WaitForOperation(sql_client, operation_ref, message)
    if is_bak_import and args.recovery_only:
        log.status.write('Bring database [{database}] online with recovery-only.\n'.format(database=args.database))
    else:
        log.status.write('Imported data from [{bucket}] into [{instance}].\n'.format(instance=instance_ref, bucket=args.uri))
    return None