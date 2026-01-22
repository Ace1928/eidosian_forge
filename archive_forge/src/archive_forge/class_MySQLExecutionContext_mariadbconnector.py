import re
from uuid import UUID as _python_UUID
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from ... import sql
from ... import util
from ...sql import sqltypes
class MySQLExecutionContext_mariadbconnector(MySQLExecutionContext):
    _lastrowid = None

    def create_server_side_cursor(self):
        return self._dbapi_connection.cursor(buffered=False)

    def create_default_cursor(self):
        return self._dbapi_connection.cursor(buffered=True)

    def post_exec(self):
        super().post_exec()
        self._rowcount = self.cursor.rowcount
        if self.isinsert and self.compiled.postfetch_lastrowid:
            self._lastrowid = self.cursor.lastrowid

    def get_lastrowid(self):
        return self._lastrowid