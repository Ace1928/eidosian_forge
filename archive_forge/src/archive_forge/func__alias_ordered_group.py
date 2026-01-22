from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _alias_ordered_group(expression: exp.Expression) -> exp.Expression:
    if isinstance(expression, exp.Select):
        group = expression.args.get('group')
        order = expression.args.get('order')
        if group and order:
            aliases = {select.this: select.args['alias'] for select in expression.selects if isinstance(select, exp.Alias)}
            for grouped in group.expressions:
                if grouped.is_int:
                    continue
                alias = aliases.get(grouped)
                if alias:
                    grouped.replace(exp.column(alias))
    return expression