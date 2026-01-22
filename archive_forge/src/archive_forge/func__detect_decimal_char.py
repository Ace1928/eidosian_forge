from __future__ import annotations
import decimal
import random
import re
from . import base as oracle
from .base import OracleCompiler
from .base import OracleDialect
from .base import OracleExecutionContext
from .types import _OracleDateLiteralRender
from ... import exc
from ... import util
from ...engine import cursor as _cursor
from ...engine import interfaces
from ...engine import processors
from ...sql import sqltypes
from ...sql._typing import is_sql_compiler
def _detect_decimal_char(self, connection):
    dbapi_connection = connection.connection
    with dbapi_connection.cursor() as cursor:

        def output_type_handler(cursor, name, defaultType, size, precision, scale):
            return cursor.var(self.dbapi.STRING, 255, arraysize=cursor.arraysize)
        cursor.outputtypehandler = output_type_handler
        cursor.execute('SELECT 1.1 FROM DUAL')
        value = cursor.fetchone()[0]
        decimal_char = value.lstrip('0')[1]
        assert not decimal_char[0].isdigit()
    self._decimal_char = decimal_char
    if self._decimal_char != '.':
        _detect_decimal = self._detect_decimal
        _to_decimal = self._to_decimal
        self._detect_decimal = lambda value: _detect_decimal(value.replace(self._decimal_char, '.'))
        self._to_decimal = lambda value: _to_decimal(value.replace(self._decimal_char, '.'))