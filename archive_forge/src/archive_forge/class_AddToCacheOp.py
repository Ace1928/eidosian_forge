from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import resource_cache
import six
class AddToCacheOp(_UpdateCacheOp):
    """An AddToCache operation."""

    def UpdateRows(self, table, rows):
        """Adds rows to table."""
        table.AddRows(rows)