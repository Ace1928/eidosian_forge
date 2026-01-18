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
def unit_to_str(expression: exp.Expression, default: str='DAY') -> t.Optional[exp.Expression]:
    unit = expression.args.get('unit')
    if isinstance(unit, exp.Placeholder):
        return unit
    if unit:
        return exp.Literal.string(unit.name)
    return exp.Literal.string(default) if default else None