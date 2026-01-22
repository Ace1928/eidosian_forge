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
def _parse_partition_bound_spec(self) -> exp.PartitionBoundSpec:

    def _parse_partition_bound_expr() -> t.Optional[exp.Expression]:
        if self._match_text_seq('MINVALUE'):
            return exp.var('MINVALUE')
        if self._match_text_seq('MAXVALUE'):
            return exp.var('MAXVALUE')
        return self._parse_bitwise()
    this: t.Optional[exp.Expression | t.List[exp.Expression]] = None
    expression = None
    from_expressions = None
    to_expressions = None
    if self._match(TokenType.IN):
        this = self._parse_wrapped_csv(self._parse_bitwise)
    elif self._match(TokenType.FROM):
        from_expressions = self._parse_wrapped_csv(_parse_partition_bound_expr)
        self._match_text_seq('TO')
        to_expressions = self._parse_wrapped_csv(_parse_partition_bound_expr)
    elif self._match_text_seq('WITH', '(', 'MODULUS'):
        this = self._parse_number()
        self._match_text_seq(',', 'REMAINDER')
        expression = self._parse_number()
        self._match_r_paren()
    else:
        self.raise_error('Failed to parse partition bound spec.')
    return self.expression(exp.PartitionBoundSpec, this=this, expression=expression, from_expressions=from_expressions, to_expressions=to_expressions)