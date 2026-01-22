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
def _parse_unnest(self, with_alias: bool=True) -> t.Optional[exp.Unnest]:
    if not self._match(TokenType.UNNEST):
        return None
    expressions = self._parse_wrapped_csv(self._parse_equality)
    offset = self._match_pair(TokenType.WITH, TokenType.ORDINALITY)
    alias = self._parse_table_alias() if with_alias else None
    if alias:
        if self.dialect.UNNEST_COLUMN_ONLY:
            if alias.args.get('columns'):
                self.raise_error('Unexpected extra column alias in unnest.')
            alias.set('columns', [alias.this])
            alias.set('this', None)
        columns = alias.args.get('columns') or []
        if offset and len(expressions) < len(columns):
            offset = columns.pop()
    if not offset and self._match_pair(TokenType.WITH, TokenType.OFFSET):
        self._match(TokenType.ALIAS)
        offset = self._parse_id_var(any_token=False, tokens=self.UNNEST_OFFSET_ALIAS_TOKENS) or exp.to_identifier('offset')
    return self.expression(exp.Unnest, expressions=expressions, alias=alias, offset=offset)