from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def transaction_sql(self, expression: exp.Transaction) -> str:
    modes = expression.args.get('modes')
    modes = f' {', '.join(modes)}' if modes else ''
    return f'START TRANSACTION{modes}'