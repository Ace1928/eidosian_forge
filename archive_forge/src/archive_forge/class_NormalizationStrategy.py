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
class NormalizationStrategy(str, AutoName):
    """Specifies the strategy according to which identifiers should be normalized."""
    LOWERCASE = auto()
    'Unquoted identifiers are lowercased.'
    UPPERCASE = auto()
    'Unquoted identifiers are uppercased.'
    CASE_SENSITIVE = auto()
    'Always case-sensitive, regardless of quotes.'
    CASE_INSENSITIVE = auto()
    'Always case-insensitive, regardless of quotes.'