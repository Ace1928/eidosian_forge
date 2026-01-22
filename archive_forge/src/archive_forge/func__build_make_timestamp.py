from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _build_make_timestamp(args: t.List) -> exp.Expression:
    if len(args) == 1:
        return exp.UnixToTime(this=seq_get(args, 0), scale=exp.UnixToTime.MICROS)
    return exp.TimestampFromParts(year=seq_get(args, 0), month=seq_get(args, 1), day=seq_get(args, 2), hour=seq_get(args, 3), min=seq_get(args, 4), sec=seq_get(args, 5))