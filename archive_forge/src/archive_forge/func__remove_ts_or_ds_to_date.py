from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _remove_ts_or_ds_to_date(to_sql: t.Optional[t.Callable[[MySQL.Generator, exp.Expression], str]]=None, args: t.Tuple[str, ...]=('this',)) -> t.Callable[[MySQL.Generator, exp.Func], str]:

    def func(self: MySQL.Generator, expression: exp.Func) -> str:
        for arg_key in args:
            arg = expression.args.get(arg_key)
            if isinstance(arg, exp.TsOrDsToDate) and (not arg.args.get('format')):
                expression.set(arg_key, arg.this)
        return to_sql(self, expression) if to_sql else self.function_fallback_sql(expression)
    return func