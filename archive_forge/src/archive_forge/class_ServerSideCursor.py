import decimal
import re
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .pg_catalog import _SpaceVector
from .pg_catalog import OIDVECTOR
from .types import CITEXT
from ... import exc
from ... import util
from ...engine import processors
from ...sql import sqltypes
from ...sql.elements import quoted_name
class ServerSideCursor:
    server_side = True

    def __init__(self, cursor, ident):
        self.ident = ident
        self.cursor = cursor

    @property
    def connection(self):
        return self.cursor.connection

    @property
    def rowcount(self):
        return self.cursor.rowcount

    @property
    def description(self):
        return self.cursor.description

    def execute(self, operation, args=(), stream=None):
        op = 'DECLARE ' + self.ident + ' NO SCROLL CURSOR FOR ' + operation
        self.cursor.execute(op, args, stream=stream)
        return self

    def executemany(self, operation, param_sets):
        self.cursor.executemany(operation, param_sets)
        return self

    def fetchone(self):
        self.cursor.execute('FETCH FORWARD 1 FROM ' + self.ident)
        return self.cursor.fetchone()

    def fetchmany(self, num=None):
        if num is None:
            return self.fetchall()
        else:
            self.cursor.execute('FETCH FORWARD ' + str(int(num)) + ' FROM ' + self.ident)
            return self.cursor.fetchall()

    def fetchall(self):
        self.cursor.execute('FETCH FORWARD ALL FROM ' + self.ident)
        return self.cursor.fetchall()

    def close(self):
        self.cursor.execute('CLOSE ' + self.ident)
        self.cursor.close()

    def setinputsizes(self, *sizes):
        self.cursor.setinputsizes(*sizes)

    def setoutputsize(self, size, column=None):
        pass