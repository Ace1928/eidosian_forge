from __future__ import annotations
import logging
import typing as t
from enum import Enum, auto
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.generator import Generator
from sqlglot.helper import AutoName, flatten, is_int, seq_get
from sqlglot.jsonpath import parse as parse_json_path
from sqlglot.parser import Parser
from sqlglot.time import TIMEZONES, format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import new_trie
def ts_or_ds_add_cast(expression: exp.TsOrDsAdd) -> exp.TsOrDsAdd:
    this = expression.this.copy()
    return_type = expression.return_type
    if return_type.is_type(exp.DataType.Type.DATE):
        this = exp.cast(this, exp.DataType.Type.TIMESTAMP)
    expression.this.replace(exp.cast(this, return_type))
    return expression