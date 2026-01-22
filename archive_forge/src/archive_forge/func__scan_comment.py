from __future__ import annotations
import os
import typing as t
from enum import auto
from sqlglot.errors import SqlglotError, TokenError
from sqlglot.helper import AutoName
from sqlglot.trie import TrieResult, in_trie, new_trie
def _scan_comment(self, comment_start: str) -> bool:
    if comment_start not in self._COMMENTS:
        return False
    comment_start_line = self._line
    comment_start_size = len(comment_start)
    comment_end = self._COMMENTS[comment_start]
    if comment_end:
        self._advance(comment_start_size)
        comment_end_size = len(comment_end)
        while not self._end and self._chars(comment_end_size) != comment_end:
            self._advance(alnum=True)
        self._comments.append(self._text[comment_start_size:-comment_end_size + 1])
        self._advance(comment_end_size - 1)
    else:
        while not self._end and self.WHITE_SPACE.get(self._peek) is not TokenType.BREAK:
            self._advance(alnum=True)
        self._comments.append(self._text[comment_start_size:])
    if comment_start_line == self._prev_token_line:
        self.tokens[-1].comments.extend(self._comments)
        self._comments = []
        self._prev_token_line = self._line
    return True