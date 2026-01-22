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
def _parse_key_constraint_options(self) -> t.List[str]:
    options = []
    while True:
        if not self._curr:
            break
        if self._match(TokenType.ON):
            action = None
            on = self._advance_any() and self._prev.text
            if self._match_text_seq('NO', 'ACTION'):
                action = 'NO ACTION'
            elif self._match_text_seq('CASCADE'):
                action = 'CASCADE'
            elif self._match_text_seq('RESTRICT'):
                action = 'RESTRICT'
            elif self._match_pair(TokenType.SET, TokenType.NULL):
                action = 'SET NULL'
            elif self._match_pair(TokenType.SET, TokenType.DEFAULT):
                action = 'SET DEFAULT'
            else:
                self.raise_error('Invalid key constraint')
            options.append(f'ON {on} {action}')
        elif self._match_text_seq('NOT', 'ENFORCED'):
            options.append('NOT ENFORCED')
        elif self._match_text_seq('DEFERRABLE'):
            options.append('DEFERRABLE')
        elif self._match_text_seq('INITIALLY', 'DEFERRED'):
            options.append('INITIALLY DEFERRED')
        elif self._match_text_seq('NORELY'):
            options.append('NORELY')
        elif self._match_text_seq('MATCH', 'FULL'):
            options.append('MATCH FULL')
        else:
            break
    return options