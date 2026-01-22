from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _returnsproperty_sql(self: BigQuery.Generator, expression: exp.ReturnsProperty) -> str:
    this = expression.this
    if isinstance(this, exp.Schema):
        this = f'{self.sql(this, 'this')} <{self.expressions(this)}>'
    else:
        this = self.sql(this)
    return f'RETURNS {this}'