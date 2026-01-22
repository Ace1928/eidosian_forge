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
def _parse_drop(self, exists: bool=False) -> exp.Drop | exp.Command:
    start = self._prev
    temporary = self._match(TokenType.TEMPORARY)
    materialized = self._match_text_seq('MATERIALIZED')
    kind = self._match_set(self.CREATABLES) and self._prev.text
    if not kind:
        return self._parse_as_command(start)
    if_exists = exists or self._parse_exists()
    table = self._parse_table_parts(schema=True, is_db_reference=self._prev.token_type == TokenType.SCHEMA)
    if self._match(TokenType.L_PAREN, advance=False):
        expressions = self._parse_wrapped_csv(self._parse_types)
    else:
        expressions = None
    return self.expression(exp.Drop, comments=start.comments, exists=if_exists, this=table, expressions=expressions, kind=kind, temporary=temporary, materialized=materialized, cascade=self._match_text_seq('CASCADE'), constraints=self._match_text_seq('CONSTRAINTS'), purge=self._match_text_seq('PURGE'))