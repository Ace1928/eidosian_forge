from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _build_datetimefromparts(args: t.List) -> exp.TimestampFromParts:
    return exp.TimestampFromParts(year=seq_get(args, 0), month=seq_get(args, 1), day=seq_get(args, 2), hour=seq_get(args, 3), min=seq_get(args, 4), sec=seq_get(args, 5), milli=seq_get(args, 6))