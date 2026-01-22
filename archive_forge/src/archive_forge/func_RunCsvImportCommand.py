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
def RunCsvImportCommand(args, client):
    """Imports data from a CSV file into Cloud SQL instance.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.
    client: SqlClient instance, with sql_client and sql_messages props, for use
      in generating messages and making API calls.

  Returns:
    A dict representing the import operation resource, if '--async' is used,
    or else None.
  """
    csv_import_context = import_util.CsvImportContext(client.sql_messages, args.uri, args.database, args.table, args.columns, args.user, args.quote, args.escape, args.fields_terminated_by, args.lines_terminated_by)
    return RunImportCommand(args, client, csv_import_context)