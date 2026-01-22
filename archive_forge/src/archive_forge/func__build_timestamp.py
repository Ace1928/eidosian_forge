from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _build_timestamp(args: t.List) -> exp.Timestamp:
    timestamp = exp.Timestamp.from_arg_list(args)
    timestamp.set('with_tz', True)
    return timestamp