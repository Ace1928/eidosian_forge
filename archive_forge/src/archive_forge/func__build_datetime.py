from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _build_datetime(name: str, kind: exp.DataType.Type, safe: bool=False) -> t.Callable[[t.List], exp.Func]:

    def _builder(args: t.List) -> exp.Func:
        value = seq_get(args, 0)
        if isinstance(value, exp.Literal):
            int_value = is_int(value.this)
            if len(args) == 1 and value.is_string and (not int_value):
                return exp.cast(value, kind)
            if kind == exp.DataType.Type.TIMESTAMP:
                if int_value:
                    return exp.UnixToTime(this=value, scale=seq_get(args, 1))
                if not is_float(value.this):
                    return build_formatted_time(exp.StrToTime, 'snowflake')(args)
        if len(args) == 2 and kind == exp.DataType.Type.DATE:
            formatted_exp = build_formatted_time(exp.TsOrDsToDate, 'snowflake')(args)
            formatted_exp.set('safe', safe)
            return formatted_exp
        return exp.Anonymous(this=name, expressions=args)
    return _builder