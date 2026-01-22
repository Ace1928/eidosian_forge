from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _parse_table_part(self, schema: bool=False) -> t.Optional[exp.Expression]:
    this = super()._parse_table_part(schema=schema) or self._parse_number()
    if isinstance(this, exp.Identifier):
        table_name = this.name
        while self._match(TokenType.DASH, advance=False) and self._next:
            text = ''
            while self._curr and self._curr.token_type != TokenType.DOT:
                self._advance()
                text += self._prev.text
            table_name += text
        this = exp.Identifier(this=table_name, quoted=this.args.get('quoted'))
    elif isinstance(this, exp.Literal):
        table_name = this.name
        if self._is_connected() and self._parse_var(any_token=True):
            table_name += self._prev.text
        this = exp.Identifier(this=table_name, quoted=True)
    return this