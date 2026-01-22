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
def RunSqlExportCommand(args, client):
    """Exports data from a Cloud SQL instance to a MySQL dump file.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.
    client: SqlClient instance, with sql_client and sql_messages props, for use
      in generating messages and making API calls.

  Returns:
    A dict object representing the operations resource describing the export
    operation if the export was successful.
  """
    sql_export_context = export_util.SqlExportContext(client.sql_messages, args.uri, args.database, args.table, offload=args.offload, parallel=args.parallel, threads=args.threads)
    if args.offload:
        log.status.write('Serverless exports cost extra. See the pricing page for more information: https://cloud.google.com/sql/pricing.\n')
    return RunExportCommand(args, client, sql_export_context)