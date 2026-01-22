from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def DisplayQueryResults(result, out):
    """Prints the result rows for a query.

  Args:
    result (spanner_v1_messages.ResultSet): The server response to a query.
    out: Output stream to which we print.
  """
    if hasattr(result.stats, 'rowCountExact') and result.stats.rowCountExact is not None:
        _DisplayNumberOfRowsModified(result.stats.rowCountExact, True, out)
    if hasattr(result.stats, 'rowCountLowerBound') and result.stats.rowCountLowerBound is not None:
        _DisplayNumberOfRowsModified(result.stats.rowCountLowerBound, False, out)
    if result.metadata.rowType.fields:
        fields = [field.name or '(Unspecified)' for field in result.metadata.rowType.fields]
        table_format = ','.join(('row.slice({0}).join():label="{1}"'.format(i, f) for i, f in enumerate(fields)))
        rows = [{'row': encoding.MessageToPyValue(row.entry)} for row in result.rows]
        resource_printer.Print(rows, 'table({0})'.format(table_format), out=out)