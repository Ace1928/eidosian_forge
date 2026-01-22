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
def _parse_set_item(self) -> t.Optional[exp.Expression]:
    parser = self._find_parser(self.SET_PARSERS, self.SET_TRIE)
    return parser(self) if parser else self._parse_set_item_assignment(kind=None)