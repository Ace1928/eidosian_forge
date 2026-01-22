import os
from typing import Dict, Type
class _PostgreSQL_dbutils(Generic_dbutils):
    """Base class for any PostgreSQL adaptor."""

    def next_id(self, cursor, table):
        table = self.tname(table)
        sql = f"SELECT nextval('{table}_pk_seq')"
        cursor.execute(sql)
        rv = cursor.fetchone()
        return rv[0]

    def last_id(self, cursor, table):
        table = self.tname(table)
        sql = f"SELECT currval('{table}_pk_seq')"
        cursor.execute(sql)
        rv = cursor.fetchone()
        return rv[0]