from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def timefromparts_sql(self, expression: exp.TimeFromParts) -> str:
    nano = expression.args.get('nano')
    if nano is not None:
        nano.pop()
        self.unsupported('Specifying nanoseconds is not supported in TIMEFROMPARTS.')
    if expression.args.get('fractions') is None:
        expression.set('fractions', exp.Literal.number(0))
    if expression.args.get('precision') is None:
        expression.set('precision', exp.Literal.number(0))
    return rename_func('TIMEFROMPARTS')(self, expression)