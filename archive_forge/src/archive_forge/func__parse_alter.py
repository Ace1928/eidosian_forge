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
def _parse_alter(self) -> exp.AlterTable | exp.Command:
    start = self._prev
    if not self._match(TokenType.TABLE):
        return self._parse_as_command(start)
    exists = self._parse_exists()
    only = self._match_text_seq('ONLY')
    this = self._parse_table(schema=True)
    if self._next:
        self._advance()
    parser = self.ALTER_PARSERS.get(self._prev.text.upper()) if self._prev else None
    if parser:
        actions = ensure_list(parser(self))
        options = self._parse_csv(self._parse_property)
        if not self._curr and actions:
            return self.expression(exp.AlterTable, this=this, exists=exists, actions=actions, only=only, options=options)
    return self._parse_as_command(start)