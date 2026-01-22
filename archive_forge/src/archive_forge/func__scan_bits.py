from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def _scan_bits(self) -> None:
    self._advance()
    value = self._extract_value()
    try:
        int(value, 2)
        self._add(TokenType.BIT_STRING, value[2:])
    except ValueError:
        self._add(TokenType.IDENTIFIER)