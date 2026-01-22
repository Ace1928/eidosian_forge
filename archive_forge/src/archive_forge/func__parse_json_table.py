from __future__ import annotations
import logging
import typing as t
from collections import defaultdict
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, concat_messages, merge_errors
from sqlglot.helper import apply_index_offset, ensure_list, seq_get
from sqlglot.time import format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import TrieResult, in_trie, new_trie
def _parse_json_table(self) -> exp.JSONTable:
    this = self._parse_format_json(self._parse_bitwise())
    path = self._match(TokenType.COMMA) and self._parse_string()
    error_handling = self._parse_on_handling('ERROR', 'ERROR', 'NULL')
    empty_handling = self._parse_on_handling('EMPTY', 'ERROR', 'NULL')
    schema = self._parse_json_schema()
    return exp.JSONTable(this=this, schema=schema, path=path, error_handling=error_handling, empty_handling=empty_handling)