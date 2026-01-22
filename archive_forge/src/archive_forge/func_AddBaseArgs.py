from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.spanner import database_sessions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.spanner import resource_args
from googlecloudsdk.command_lib.spanner import sql
from googlecloudsdk.command_lib.spanner.sql import QueryHasDml
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def AddBaseArgs(parser):
    """Parses provided arguments to add base arguments used for both Beta and GA.

  Args:
    parser: an argparse argument parser.
  """
    resource_args.AddDatabaseResourceArg(parser, 'to execute the SQL query against')
    parser.add_argument('--sql', required=True, help='The SQL query to issue to the database. Cloud Spanner SQL is described at https://cloud.google.com/spanner/docs/query-syntax')
    query_mode_choices = {'NORMAL': 'Returns only the query result, without any information about the query plan.', 'PLAN': 'Returns only the query plan, without any result rows or execution statistics information.', 'PROFILE': 'Returns both the query plan and the execution statistics along with the result rows.'}
    parser.add_argument('--query-mode', default='NORMAL', type=lambda x: x.upper(), choices=query_mode_choices, help='Mode in which the query must be processed.')
    parser.add_argument('--enable-partitioned-dml', action='store_true', help='Execute DML statement using Partitioned DML')
    parser.add_argument('--timeout', type=arg_parsers.Duration(), default='10m', help='Maximum time to wait for the SQL query to complete. See $ gcloud topic datetimes for information on duration formats.')
    msgs = apis.GetMessagesModule('spanner', 'v1')
    GetRequestPriorityMapper(msgs).choice_arg.AddToParser(parser)
    timestamp_bound_group = parser.add_argument_group(mutex=True, help='Read-only query timestamp bound. The default is --strong. See https://cloud.google.com/spanner/docs/timestamp-bounds.')
    timestamp_bound_group.add_argument('--strong', action='store_true', help='Perform a strong query.')
    timestamp_bound_group.add_argument('--read-timestamp', metavar='TIMESTAMP', help='Perform a query at the given timestamp.')
    parser.add_argument('--database-role', help='Database role user assumes while accessing the database.')