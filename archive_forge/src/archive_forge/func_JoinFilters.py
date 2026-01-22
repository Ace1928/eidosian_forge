from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def JoinFilters(clauses, operator='AND'):
    """Join the clauses with the operator.

  This function surrounds each clause with a set of parentheses and joins the
  clauses with the operator.

  Args:
    clauses: List of strings. Each string is a clause in the filter.
    operator: Logical operator used to join the clauses

  Returns:
    The clauses joined by the operator.
  """
    return (' ' + operator + ' ').join(clauses)