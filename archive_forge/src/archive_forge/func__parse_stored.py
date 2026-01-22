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
def _parse_stored(self) -> exp.FileFormatProperty:
    self._match(TokenType.ALIAS)
    input_format = self._parse_string() if self._match_text_seq('INPUTFORMAT') else None
    output_format = self._parse_string() if self._match_text_seq('OUTPUTFORMAT') else None
    return self.expression(exp.FileFormatProperty, this=self.expression(exp.InputOutputFormat, input_format=input_format, output_format=output_format) if input_format or output_format else self._parse_var_or_string() or self._parse_number() or self._parse_id_var())