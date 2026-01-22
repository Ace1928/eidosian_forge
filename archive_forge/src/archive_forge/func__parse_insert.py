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
def _parse_insert(self) -> exp.Insert:
    comments = ensure_list(self._prev_comments)
    hint = self._parse_hint()
    overwrite = self._match(TokenType.OVERWRITE)
    ignore = self._match(TokenType.IGNORE)
    local = self._match_text_seq('LOCAL')
    alternative = None
    is_function = None
    if self._match_text_seq('DIRECTORY'):
        this: t.Optional[exp.Expression] = self.expression(exp.Directory, this=self._parse_var_or_string(), local=local, row_format=self._parse_row_format(match_row=True))
    else:
        if self._match(TokenType.OR):
            alternative = self._match_texts(self.INSERT_ALTERNATIVES) and self._prev.text
        self._match(TokenType.INTO)
        comments += ensure_list(self._prev_comments)
        self._match(TokenType.TABLE)
        is_function = self._match(TokenType.FUNCTION)
        this = self._parse_table(schema=True, parse_partition=True) if not is_function else self._parse_function()
    returning = self._parse_returning()
    return self.expression(exp.Insert, comments=comments, hint=hint, is_function=is_function, this=this, stored=self._match_text_seq('STORED') and self._parse_stored(), by_name=self._match_text_seq('BY', 'NAME'), exists=self._parse_exists(), where=self._match_pair(TokenType.REPLACE, TokenType.WHERE) and self._parse_conjunction(), expression=self._parse_derived_table_values() or self._parse_ddl_select(), conflict=self._parse_on_conflict(), returning=returning or self._parse_returning(), overwrite=overwrite, alternative=alternative, ignore=ignore)