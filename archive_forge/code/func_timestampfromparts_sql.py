from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def timestampfromparts_sql(self, expression: exp.TimestampFromParts) -> str:
    zone = expression.args.get('zone')
    if zone is not None:
        zone.pop()
        self.unsupported('Time zone is not supported in DATETIMEFROMPARTS.')
    nano = expression.args.get('nano')
    if nano is not None:
        nano.pop()
        self.unsupported('Specifying nanoseconds is not supported in DATETIMEFROMPARTS.')
    if expression.args.get('milli') is None:
        expression.set('milli', exp.Literal.number(0))
    return rename_func('DATETIMEFROMPARTS')(self, expression)