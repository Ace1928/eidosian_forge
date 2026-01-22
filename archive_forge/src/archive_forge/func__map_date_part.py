from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _map_date_part(part):
    mapped = DATE_PART_MAPPING.get(part.name.upper()) if part else None
    return exp.var(mapped) if mapped else part