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
def _parse_load(self) -> exp.LoadData | exp.Command:
    if self._match_text_seq('DATA'):
        local = self._match_text_seq('LOCAL')
        self._match_text_seq('INPATH')
        inpath = self._parse_string()
        overwrite = self._match(TokenType.OVERWRITE)
        self._match_pair(TokenType.INTO, TokenType.TABLE)
        return self.expression(exp.LoadData, this=self._parse_table(schema=True), local=local, overwrite=overwrite, inpath=inpath, partition=self._parse_partition(), input_format=self._match_text_seq('INPUTFORMAT') and self._parse_string(), serde=self._match_text_seq('SERDE') and self._parse_string())
    return self._parse_as_command(self._prev)