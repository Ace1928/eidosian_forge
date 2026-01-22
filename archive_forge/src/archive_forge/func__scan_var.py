from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def _scan_var(self) -> None:
    while True:
        char = self._peek.strip()
        if char and (char in self.VAR_SINGLE_TOKENS or char not in self.SINGLE_TOKENS):
            self._advance(alnum=True)
        else:
            break
    self._add(TokenType.VAR if self.tokens and self.tokens[-1].token_type == TokenType.PARAMETER else self.KEYWORDS.get(self._text.upper(), TokenType.VAR))