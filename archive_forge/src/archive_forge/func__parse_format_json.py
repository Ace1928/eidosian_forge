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
def _parse_format_json(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
    if not this or not self._match_text_seq('FORMAT', 'JSON'):
        return this
    return self.expression(exp.FormatJson, this=this)