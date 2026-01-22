from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def DisplayQueryAggregateStats(query_stats, out):
    """Displays the aggregate stats for a Spanner SQL query.

  Looks at the queryStats portion of the query response and prints some of
  the aggregate statistics.

  Args:
    query_stats (spanner_v1_messages.ResultSetStats.QueryStatsValue): The query
      stats taken from the server response to a query.
    out: Output stream to which we print.
  """
    get_prop = partial(_GetAdditionalProperty, query_stats.additionalProperties)
    stats = {'total_elapsed_time': _ConvertToStringValue(get_prop('elapsed_time')), 'cpu_time': _ConvertToStringValue(get_prop('cpu_time')), 'rows_returned': _ConvertToStringValue(get_prop('rows_returned')), 'rows_scanned': _ConvertToStringValue(get_prop('rows_scanned')), 'optimizer_version': _ConvertToStringValue(get_prop('optimizer_version'))}
    resource_printer.Print(stats, 'table[box](total_elapsed_time, cpu_time, rows_returned, rows_scanned, optimizer_version)', out=out)