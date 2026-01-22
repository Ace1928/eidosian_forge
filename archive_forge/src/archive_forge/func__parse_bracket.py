from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _parse_bracket(self, this: t.Optional[exp.Expression]=None) -> t.Optional[exp.Expression]:
    bracket = super()._parse_bracket(this)
    if this is bracket:
        return bracket
    if isinstance(bracket, exp.Bracket):
        for expression in bracket.expressions:
            name = expression.name.upper()
            if name not in self.BRACKET_OFFSETS:
                break
            offset, safe = self.BRACKET_OFFSETS[name]
            bracket.set('offset', offset)
            bracket.set('safe', safe)
            expression.replace(expression.expressions[0])
    return bracket