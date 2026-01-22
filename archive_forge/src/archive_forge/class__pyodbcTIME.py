import re
from .base import MySQLDialect
from .base import MySQLExecutionContext
from .types import TIME
from ... import exc
from ... import util
from ...connectors.pyodbc import PyODBCConnector
from ...sql.sqltypes import Time
class _pyodbcTIME(TIME):

    def result_processor(self, dialect, coltype):

        def process(value):
            return value
        return process