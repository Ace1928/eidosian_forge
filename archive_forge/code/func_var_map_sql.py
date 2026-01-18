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
def var_map_sql(self: Generator, expression: exp.Map | exp.VarMap, map_func_name: str='MAP') -> str:
    keys = expression.args['keys']
    values = expression.args['values']
    if not isinstance(keys, exp.Array) or not isinstance(values, exp.Array):
        self.unsupported('Cannot convert array columns into map.')
        return self.func(map_func_name, keys, values)
    args = []
    for key, value in zip(keys.expressions, values.expressions):
        args.append(self.sql(key))
        args.append(self.sql(value))
    return self.func(map_func_name, *args)