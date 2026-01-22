from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _build_to_hex(args: t.List) -> exp.Hex | exp.MD5:
    arg = seq_get(args, 0)
    return exp.MD5(this=arg.this) if isinstance(arg, exp.MD5Digest) else exp.Hex(this=arg)