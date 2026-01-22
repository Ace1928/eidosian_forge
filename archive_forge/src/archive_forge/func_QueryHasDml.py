from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def QueryHasDml(sql):
    """Determines if the sql string contains a DML query.

  Args:
    sql (string): The sql string entered by the user.

  Returns:
    A boolean.
  """
    sql = sql.lstrip().lower()
    tokenized = lexer.tokenize(sql)
    for token in list(tokenized):
        has_dml = token == (T.Keyword.DML, 'insert') or token == (T.Keyword.DML, 'update') or token == (T.Keyword.DML, 'delete')
        if has_dml:
            return True
    return False