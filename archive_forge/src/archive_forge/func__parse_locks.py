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
def _parse_locks(self) -> t.List[exp.Lock]:
    locks = []
    while True:
        if self._match_text_seq('FOR', 'UPDATE'):
            update = True
        elif self._match_text_seq('FOR', 'SHARE') or self._match_text_seq('LOCK', 'IN', 'SHARE', 'MODE'):
            update = False
        else:
            break
        expressions = None
        if self._match_text_seq('OF'):
            expressions = self._parse_csv(lambda: self._parse_table(schema=True))
        wait: t.Optional[bool | exp.Expression] = None
        if self._match_text_seq('NOWAIT'):
            wait = True
        elif self._match_text_seq('WAIT'):
            wait = self._parse_primary()
        elif self._match_text_seq('SKIP', 'LOCKED'):
            wait = False
        locks.append(self.expression(exp.Lock, update=update, expressions=expressions, wait=wait))
    return locks