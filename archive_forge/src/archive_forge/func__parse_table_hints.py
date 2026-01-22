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
def _parse_table_hints(self) -> t.Optional[t.List[exp.Expression]]:
    hints: t.List[exp.Expression] = []
    if self._match_pair(TokenType.WITH, TokenType.L_PAREN):
        hints.append(self.expression(exp.WithTableHint, expressions=self._parse_csv(lambda: self._parse_function() or self._parse_var(any_token=True))))
        self._match_r_paren()
    else:
        while self._match_set(self.TABLE_INDEX_HINT_TOKENS):
            hint = exp.IndexTableHint(this=self._prev.text.upper())
            self._match_texts(('INDEX', 'KEY'))
            if self._match(TokenType.FOR):
                hint.set('target', self._advance_any() and self._prev.text.upper())
            hint.set('expressions', self._parse_wrapped_id_vars())
            hints.append(hint)
    return hints or None