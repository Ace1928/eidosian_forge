from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlImportOptionsValue(_messages.Message):
    """Optional. Options for importing data from SQL statements.

    Fields:
      parallel: Optional. Whether or not the import should be parallel.
      threads: Optional. The number of threads to use for parallel import.
    """
    parallel = _messages.BooleanField(1)
    threads = _messages.IntegerField(2, variant=_messages.Variant.INT32)