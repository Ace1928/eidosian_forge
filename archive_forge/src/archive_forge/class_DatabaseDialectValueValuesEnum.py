from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseDialectValueValuesEnum(_messages.Enum):
    """Output only. The dialect of the Cloud Spanner Database.

    Values:
      DATABASE_DIALECT_UNSPECIFIED: Default value. This value will create a
        database with the GOOGLE_STANDARD_SQL dialect.
      GOOGLE_STANDARD_SQL: GoogleSQL supported SQL.
      POSTGRESQL: PostgreSQL supported SQL.
    """
    DATABASE_DIALECT_UNSPECIFIED = 0
    GOOGLE_STANDARD_SQL = 1
    POSTGRESQL = 2