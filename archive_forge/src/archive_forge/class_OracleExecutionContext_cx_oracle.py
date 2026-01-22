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
class OracleExecutionContext_cx_oracle(OracleExecutionContext):
    out_parameters = None

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

    def _generate_cursor_outputtype_handler(self):
        output_handlers = {}
        for keyname, name, objects, type_ in self.compiled._result_columns:
            handler = type_._cached_custom_processor(self.dialect, 'cx_oracle_outputtypehandler', self._get_cx_oracle_type_handler)
            if handler:
                denormalized_name = self.dialect.denormalize_name(keyname)
                output_handlers[denormalized_name] = handler
        if output_handlers:
            default_handler = self._dbapi_connection.outputtypehandler

            def output_type_handler(cursor, name, default_type, size, precision, scale):
                if name in output_handlers:
                    return output_handlers[name](cursor, name, default_type, size, precision, scale)
                else:
                    return default_handler(cursor, name, default_type, size, precision, scale)
            self.cursor.outputtypehandler = output_type_handler

    def _get_cx_oracle_type_handler(self, impl):
        if hasattr(impl, '_cx_oracle_outputtypehandler'):
            return impl._cx_oracle_outputtypehandler(self.dialect)
        else:
            return None

    def pre_exec(self):
        super().pre_exec()
        if not getattr(self.compiled, '_oracle_cx_sql_compiler', False):
            return
        self.out_parameters = {}
        self._generate_out_parameter_vars()
        self._generate_cursor_outputtype_handler()

    def post_exec(self):
        if self.compiled and is_sql_compiler(self.compiled) and self.compiled._oracle_returning:
            initial_buffer = self.fetchall_for_returning(self.cursor, _internal=True)
            fetch_strategy = _cursor.FullyBufferedCursorFetchStrategy(self.cursor, [(entry.keyname, None) for entry in self.compiled._result_columns], initial_buffer=initial_buffer)
            self.cursor_fetch_strategy = fetch_strategy

    def create_cursor(self):
        c = self._dbapi_connection.cursor()
        if self.dialect.arraysize:
            c.arraysize = self.dialect.arraysize
        return c

    def fetchall_for_returning(self, cursor, *, _internal=False):
        compiled = self.compiled
        if not _internal and compiled is None or not is_sql_compiler(compiled) or (not compiled._oracle_returning):
            raise NotImplementedError('execution context was not prepared for Oracle RETURNING')
        numcols = len(self.out_parameters)
        return list(zip(*[[val for stmt_result in self.out_parameters[f'ret_{j}'].values for val in stmt_result or ()] for j in range(numcols)]))

    def get_out_parameter_values(self, out_param_names):
        assert not self.compiled.returning
        return [self.dialect._paramval(self.out_parameters[name]) for name in out_param_names]