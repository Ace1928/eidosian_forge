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
def _parse_not_constraint(self) -> t.Optional[exp.Expression]:
    if self._match_text_seq('NULL'):
        return self.expression(exp.NotNullColumnConstraint)
    if self._match_text_seq('CASESPECIFIC'):
        return self.expression(exp.CaseSpecificColumnConstraint, not_=True)
    if self._match_text_seq('FOR', 'REPLICATION'):
        return self.expression(exp.NotForReplicationColumnConstraint)
    return None