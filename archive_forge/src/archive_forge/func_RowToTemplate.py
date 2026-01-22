from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import resource_cache
import six
def RowToTemplate(self, row):
    """Returns the row template of row for the Resource.Complete method.

    By default all parameters are treated as prefixes.

    Args:
      row: The resource parameter tuple.

    Returns:
      The row template of row for the Resource.Complete method.
    """
    row_template = list(row)
    if len(row) < self.columns:
        row_template += [''] * (self.columns - len(row))
    return [c if '*' in c else c + '*' for c in row_template]