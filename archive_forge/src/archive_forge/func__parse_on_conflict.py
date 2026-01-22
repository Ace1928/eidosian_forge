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
def _parse_on_conflict(self) -> t.Optional[exp.OnConflict]:
    conflict = self._match_text_seq('ON', 'CONFLICT')
    duplicate = self._match_text_seq('ON', 'DUPLICATE', 'KEY')
    if not conflict and (not duplicate):
        return None
    conflict_keys = None
    constraint = None
    if conflict:
        if self._match_text_seq('ON', 'CONSTRAINT'):
            constraint = self._parse_id_var()
        elif self._match(TokenType.L_PAREN):
            conflict_keys = self._parse_csv(self._parse_id_var)
            self._match_r_paren()
    action = self._parse_var_from_options(self.CONFLICT_ACTIONS)
    if self._prev.token_type == TokenType.UPDATE:
        self._match(TokenType.SET)
        expressions = self._parse_csv(self._parse_equality)
    else:
        expressions = None
    return self.expression(exp.OnConflict, duplicate=duplicate, expressions=expressions, action=action, conflict_keys=conflict_keys, constraint=constraint)