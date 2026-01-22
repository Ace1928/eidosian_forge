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
def _parse_index_params(self) -> exp.IndexParameters:
    using = self._parse_var(any_token=True) if self._match(TokenType.USING) else None
    if self._match(TokenType.L_PAREN, advance=False):
        columns = self._parse_wrapped_csv(self._parse_with_operator)
    else:
        columns = None
    include = self._parse_wrapped_id_vars() if self._match_text_seq('INCLUDE') else None
    partition_by = self._parse_partition_by()
    with_storage = self._match(TokenType.WITH) and self._parse_wrapped_properties()
    tablespace = self._parse_var(any_token=True) if self._match_text_seq('USING', 'INDEX', 'TABLESPACE') else None
    where = self._parse_where()
    return self.expression(exp.IndexParameters, using=using, columns=columns, include=include, partition_by=partition_by, where=where, with_storage=with_storage, tablespace=tablespace)