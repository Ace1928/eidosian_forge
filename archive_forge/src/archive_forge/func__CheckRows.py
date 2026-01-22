from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import time
from googlecloudsdk.core.cache import exceptions
import six
import six.moves.urllib.parse
def _CheckRows(self, rows):
    """Raise an exception if the size of any row in rows is invalid.

    Each row size must be equal to the number of columns in the table.

    Args:
      rows: The list of rows to check.

    Raises:
      CacheTableRowSizeInvalid: If any row has an invalid size.
    """
    for row in rows:
        if len(row) != self.columns:
            raise exceptions.CacheTableRowSizeInvalid('Cache table [{}] row size [{}] is invalid. Must be {}.'.format(self.name, len(row), self.columns))