import os
from typing import Dict, Type
class Pgdb_dbutils(_PostgreSQL_dbutils):
    """Custom database utilities for Pgdb (aka PyGreSQL, for PostgreSQL)."""

    def autocommit(self, conn, y=True):
        """Set autocommit on the database connection. Currently not implemented."""
        raise NotImplementedError('pgdb does not support this!')