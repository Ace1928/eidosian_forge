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
def arrow_json_extract_sql(self: Generator, expression: JSON_EXTRACT_TYPE) -> str:
    this = expression.this
    if self.JSON_TYPE_REQUIRED_FOR_EXTRACTION and isinstance(this, exp.Literal) and this.is_string:
        this.replace(exp.cast(this, exp.DataType.Type.JSON))
    return self.binary(expression, '->' if isinstance(expression, exp.JSONExtract) else '->>')