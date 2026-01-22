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
def _parse_json_column_def(self) -> exp.JSONColumnDef:
    if not self._match_text_seq('NESTED'):
        this = self._parse_id_var()
        kind = self._parse_types(allow_identifiers=False)
        nested = None
    else:
        this = None
        kind = None
        nested = True
    path = self._match_text_seq('PATH') and self._parse_string()
    nested_schema = nested and self._parse_json_schema()
    return self.expression(exp.JSONColumnDef, this=this, kind=kind, path=path, nested_schema=nested_schema)