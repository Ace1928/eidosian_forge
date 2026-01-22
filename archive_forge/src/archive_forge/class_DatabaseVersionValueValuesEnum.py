from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseVersionValueValuesEnum(_messages.Enum):
    """The database engine type and version.

    Values:
      SQL_DATABASE_VERSION_UNSPECIFIED: Unspecified version.
      MYSQL_5_6: MySQL 5.6.
      MYSQL_5_7: MySQL 5.7.
      MYSQL_8_0: MySQL 8.0.
    """
    SQL_DATABASE_VERSION_UNSPECIFIED = 0
    MYSQL_5_6 = 1
    MYSQL_5_7 = 2
    MYSQL_8_0 = 3