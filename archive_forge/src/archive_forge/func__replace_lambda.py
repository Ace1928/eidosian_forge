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
def _replace_lambda(self, node: t.Optional[exp.Expression], lambda_variables: t.Set[str]) -> t.Optional[exp.Expression]:
    if not node:
        return node
    for column in node.find_all(exp.Column):
        if column.parts[0].name in lambda_variables:
            dot_or_id = column.to_dot() if column.table else column.this
            parent = column.parent
            while isinstance(parent, exp.Dot):
                if not isinstance(parent.parent, exp.Dot):
                    parent.replace(dot_or_id)
                    break
                parent = parent.parent
            else:
                if column is node:
                    node = dot_or_id
                else:
                    column.replace(dot_or_id)
    return node