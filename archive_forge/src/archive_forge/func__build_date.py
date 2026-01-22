from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _build_date(args: t.List) -> exp.Date | exp.DateFromParts:
    expr_type = exp.DateFromParts if len(args) == 3 else exp.Date
    return expr_type.from_arg_list(args)