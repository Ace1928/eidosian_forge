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
def _parse_table_alias(self, alias_tokens: t.Optional[t.Collection[TokenType]]=None) -> t.Optional[exp.TableAlias]:
    any_token = self._match(TokenType.ALIAS)
    alias = self._parse_id_var(any_token=any_token, tokens=alias_tokens or self.TABLE_ALIAS_TOKENS) or self._parse_string_as_identifier()
    index = self._index
    if self._match(TokenType.L_PAREN):
        columns = self._parse_csv(self._parse_function_parameter)
        self._match_r_paren() if columns else self._retreat(index)
    else:
        columns = None
    if not alias and (not columns):
        return None
    return self.expression(exp.TableAlias, this=alias, columns=columns)