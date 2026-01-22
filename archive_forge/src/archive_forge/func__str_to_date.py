from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _str_to_date(args: t.List) -> exp.StrToDate | exp.StrToTime:
    mysql_date_format = seq_get(args, 1)
    date_format = MySQL.format_time(mysql_date_format)
    this = seq_get(args, 0)
    if mysql_date_format and _has_time_specifier(mysql_date_format.name):
        return exp.StrToTime(this=this, format=date_format)
    return exp.StrToDate(this=this, format=date_format)