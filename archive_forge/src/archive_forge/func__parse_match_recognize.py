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
def _parse_match_recognize(self) -> t.Optional[exp.MatchRecognize]:
    if not self._match(TokenType.MATCH_RECOGNIZE):
        return None
    self._match_l_paren()
    partition = self._parse_partition_by()
    order = self._parse_order()
    measures = self._parse_csv(self._parse_match_recognize_measure) if self._match_text_seq('MEASURES') else None
    if self._match_text_seq('ONE', 'ROW', 'PER', 'MATCH'):
        rows = exp.var('ONE ROW PER MATCH')
    elif self._match_text_seq('ALL', 'ROWS', 'PER', 'MATCH'):
        text = 'ALL ROWS PER MATCH'
        if self._match_text_seq('SHOW', 'EMPTY', 'MATCHES'):
            text += ' SHOW EMPTY MATCHES'
        elif self._match_text_seq('OMIT', 'EMPTY', 'MATCHES'):
            text += ' OMIT EMPTY MATCHES'
        elif self._match_text_seq('WITH', 'UNMATCHED', 'ROWS'):
            text += ' WITH UNMATCHED ROWS'
        rows = exp.var(text)
    else:
        rows = None
    if self._match_text_seq('AFTER', 'MATCH', 'SKIP'):
        text = 'AFTER MATCH SKIP'
        if self._match_text_seq('PAST', 'LAST', 'ROW'):
            text += ' PAST LAST ROW'
        elif self._match_text_seq('TO', 'NEXT', 'ROW'):
            text += ' TO NEXT ROW'
        elif self._match_text_seq('TO', 'FIRST'):
            text += f' TO FIRST {self._advance_any().text}'
        elif self._match_text_seq('TO', 'LAST'):
            text += f' TO LAST {self._advance_any().text}'
        after = exp.var(text)
    else:
        after = None
    if self._match_text_seq('PATTERN'):
        self._match_l_paren()
        if not self._curr:
            self.raise_error('Expecting )', self._curr)
        paren = 1
        start = self._curr
        while self._curr and paren > 0:
            if self._curr.token_type == TokenType.L_PAREN:
                paren += 1
            if self._curr.token_type == TokenType.R_PAREN:
                paren -= 1
            end = self._prev
            self._advance()
        if paren > 0:
            self.raise_error('Expecting )', self._curr)
        pattern = exp.var(self._find_sql(start, end))
    else:
        pattern = None
    define = self._parse_csv(self._parse_name_as_expression) if self._match_text_seq('DEFINE') else None
    self._match_r_paren()
    return self.expression(exp.MatchRecognize, partition_by=partition, order=order, measures=measures, rows=rows, after=after, pattern=pattern, define=define, alias=self._parse_table_alias())