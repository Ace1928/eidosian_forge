from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _build_timefromparts(args: t.List) -> exp.TimeFromParts:
    return exp.TimeFromParts(hour=seq_get(args, 0), min=seq_get(args, 1), sec=seq_get(args, 2), fractions=seq_get(args, 3), precision=seq_get(args, 4))