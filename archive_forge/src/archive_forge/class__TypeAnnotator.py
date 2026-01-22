from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
class _TypeAnnotator(type):

    def __new__(cls, clsname, bases, attrs):
        klass = super().__new__(cls, clsname, bases, attrs)
        text_precedence = (exp.DataType.Type.TEXT, exp.DataType.Type.NVARCHAR, exp.DataType.Type.VARCHAR, exp.DataType.Type.NCHAR, exp.DataType.Type.CHAR)
        numeric_precedence = (exp.DataType.Type.DOUBLE, exp.DataType.Type.FLOAT, exp.DataType.Type.DECIMAL, exp.DataType.Type.BIGINT, exp.DataType.Type.INT, exp.DataType.Type.SMALLINT, exp.DataType.Type.TINYINT)
        timelike_precedence = (exp.DataType.Type.TIMESTAMPLTZ, exp.DataType.Type.TIMESTAMPTZ, exp.DataType.Type.TIMESTAMP, exp.DataType.Type.DATETIME, exp.DataType.Type.DATE)
        for type_precedence in (text_precedence, numeric_precedence, timelike_precedence):
            coerces_to = set()
            for data_type in type_precedence:
                klass.COERCES_TO[data_type] = coerces_to.copy()
                coerces_to |= {data_type}
        return klass