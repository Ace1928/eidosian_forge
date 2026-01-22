from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseDumpTypeValueValuesEnum(_messages.Enum):
    """Output only. The type of the database dump.

    Values:
      TYPE_UNSPECIFIED: The type of the database dump is unknown.
      MYSQL: Database dump is a MySQL dump file.
      AVRO: Database dump contains Avro files.
    """
    TYPE_UNSPECIFIED = 0
    MYSQL = 1
    AVRO = 2