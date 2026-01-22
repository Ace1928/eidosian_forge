from .base import BIT
from .base import MySQLDialect
from .mysqldb import MySQLDialect_mysqldb
from ... import util
class _cymysqlBIT(BIT):

    def result_processor(self, dialect, coltype):
        """Convert MySQL's 64 bit, variable length binary string to a long."""

        def process(value):
            if value is not None:
                v = 0
                for i in iter(value):
                    v = v << 8 | i
                return v
            return value
        return process