from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _build_time(args: t.List) -> exp.Func:
    if len(args) == 1:
        return exp.TsOrDsToTime(this=args[0])
    if len(args) == 3:
        return exp.TimeFromParts.from_arg_list(args)
    return exp.Anonymous(this='TIME', expressions=args)