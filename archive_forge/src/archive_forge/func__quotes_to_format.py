from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def _quotes_to_format(token_type: TokenType, arr: t.List[str | t.Tuple[str, str]]) -> t.Dict[str, t.Tuple[str, TokenType]]:
    return {k: (v, token_type) for k, v in _convert_quotes(arr).items()}