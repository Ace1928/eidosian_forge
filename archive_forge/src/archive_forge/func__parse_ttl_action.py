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
def _parse_ttl_action() -> t.Optional[exp.Expression]:
    this = self._parse_bitwise()
    if self._match_text_seq('DELETE'):
        return self.expression(exp.MergeTreeTTLAction, this=this, delete=True)
    if self._match_text_seq('RECOMPRESS'):
        return self.expression(exp.MergeTreeTTLAction, this=this, recompress=self._parse_bitwise())
    if self._match_text_seq('TO', 'DISK'):
        return self.expression(exp.MergeTreeTTLAction, this=this, to_disk=self._parse_string())
    if self._match_text_seq('TO', 'VOLUME'):
        return self.expression(exp.MergeTreeTTLAction, this=this, to_volume=self._parse_string())
    return this