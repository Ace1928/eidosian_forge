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
def _generate_out_parameter_vars(self):
    if self.compiled.has_out_parameters or self.compiled._oracle_returning:
        out_parameters = self.out_parameters
        assert out_parameters is not None
        len_params = len(self.parameters)
        quoted_bind_names = self.compiled.escaped_bind_names
        for bindparam in self.compiled.binds.values():
            if bindparam.isoutparam:
                name = self.compiled.bind_names[bindparam]
                type_impl = bindparam.type.dialect_impl(self.dialect)
                if hasattr(type_impl, '_cx_oracle_var'):
                    out_parameters[name] = type_impl._cx_oracle_var(self.dialect, self.cursor, arraysize=len_params)
                else:
                    dbtype = type_impl.get_dbapi_type(self.dialect.dbapi)
                    cx_Oracle = self.dialect.dbapi
                    assert cx_Oracle is not None
                    if dbtype is None:
                        raise exc.InvalidRequestError('Cannot create out parameter for parameter %r - its type %r is not supported by cx_oracle' % (bindparam.key, bindparam.type))
                    if isinstance(type_impl, _LOBDataType):
                        if dbtype == cx_Oracle.DB_TYPE_NVARCHAR:
                            dbtype = cx_Oracle.NCLOB
                        elif dbtype == cx_Oracle.DB_TYPE_RAW:
                            dbtype = cx_Oracle.BLOB
                        out_parameters[name] = self.cursor.var(dbtype, outconverter=lambda value: value.read(), arraysize=len_params)
                    elif isinstance(type_impl, _OracleNumeric) and type_impl.asdecimal:
                        out_parameters[name] = self.cursor.var(decimal.Decimal, arraysize=len_params)
                    else:
                        out_parameters[name] = self.cursor.var(dbtype, arraysize=len_params)
                for param in self.parameters:
                    param[quoted_bind_names.get(name, name)] = out_parameters[name]