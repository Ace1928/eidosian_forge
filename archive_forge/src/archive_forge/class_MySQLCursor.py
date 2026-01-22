import sys
import numpy as np
from pymysql import connect
from pymysql.err import ProgrammingError
from copy import deepcopy
from ase.db.sqlite import SQLite3Database
from ase.db.sqlite import init_statements
from ase.db.sqlite import VERSION
from ase.db.postgresql import remove_nan_and_inf, insert_nan_and_inf
import ase.io.jsonio
import json
class MySQLCursor:
    """
    Wrapper for the MySQL cursor. The most important task performed by this
    class is to translate SQLite queries to MySQL. Translation is needed
    because ASE DB uses some field names that are reserved words in MySQL.
    Thus, these has to mapped onto other field names.
    """
    sql_replace = [(' key TEXT', ' attribute_key TEXT'), ('(key TEXT', '(attribute_key TEXT'), ('SELECT key FROM', 'SELECT attribute_key FROM'), ('?', '%s'), (' keys ', ' attribute_keys '), (' key=', ' attribute_key='), ('table.key', 'table.attribute_key'), (' IF NOT EXISTS', '')]

    def __init__(self, cur):
        self.cur = cur

    def execute(self, sql, params=None):
        for substibution in self.sql_replace:
            sql = sql.replace(substibution[0], substibution[1])
        if params is None:
            params = ()
        self.cur.execute(sql, params)

    def fetchone(self):
        return self.cur.fetchone()

    def fetchall(self):
        return self.cur.fetchall()

    def _replace_nan_inf_kvp(self, values):
        for item in values:
            if not np.isfinite(item[1]):
                item[1] = sys.float_info.max / 2
        return values

    def executemany(self, sql, values):
        if 'number_key_values' in sql:
            values = self._replace_nan_inf_kvp(values)
        for substibution in self.sql_replace:
            sql = sql.replace(substibution[0], substibution[1])
        self.cur.executemany(sql, values)