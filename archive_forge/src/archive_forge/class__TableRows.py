from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.cache import resource_cache
import six
class _TableRows(object):
    """An _UpdateCacheOp._GetTablesFromUris dict entry."""

    def __init__(self, table):
        self.table = table
        self.rows = []