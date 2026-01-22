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
def _parse_row_format(self, match_row: bool=False) -> t.Optional[exp.RowFormatSerdeProperty | exp.RowFormatDelimitedProperty]:
    if match_row and (not self._match_pair(TokenType.ROW, TokenType.FORMAT)):
        return None
    if self._match_text_seq('SERDE'):
        this = self._parse_string()
        serde_properties = None
        if self._match(TokenType.SERDE_PROPERTIES):
            serde_properties = self.expression(exp.SerdeProperties, expressions=self._parse_wrapped_properties())
        return self.expression(exp.RowFormatSerdeProperty, this=this, serde_properties=serde_properties)
    self._match_text_seq('DELIMITED')
    kwargs = {}
    if self._match_text_seq('FIELDS', 'TERMINATED', 'BY'):
        kwargs['fields'] = self._parse_string()
        if self._match_text_seq('ESCAPED', 'BY'):
            kwargs['escaped'] = self._parse_string()
    if self._match_text_seq('COLLECTION', 'ITEMS', 'TERMINATED', 'BY'):
        kwargs['collection_items'] = self._parse_string()
    if self._match_text_seq('MAP', 'KEYS', 'TERMINATED', 'BY'):
        kwargs['map_keys'] = self._parse_string()
    if self._match_text_seq('LINES', 'TERMINATED', 'BY'):
        kwargs['lines'] = self._parse_string()
    if self._match_text_seq('NULL', 'DEFINED', 'AS'):
        kwargs['null'] = self._parse_string()
    return self.expression(exp.RowFormatDelimitedProperty, **kwargs)