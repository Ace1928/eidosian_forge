from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def QueryHasAggregateStats(result):
    """Checks if the given results have aggregate statistics.

  Args:
    result (spanner_v1_messages.ResultSetStats): The stats for a query.

  Returns:
    A boolean indicating whether 'results' contain aggregate statistics.
  """
    return hasattr(result, 'stats') and getattr(result.stats, 'queryStats', None) is not None