from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _build_timetostr_or_tochar(args: t.List) -> exp.TimeToStr | exp.ToChar:
    this = seq_get(args, 0)
    if this and (not this.type):
        from sqlglot.optimizer.annotate_types import annotate_types
        annotate_types(this)
        if this.is_type(*exp.DataType.TEMPORAL_TYPES):
            return build_formatted_time(exp.TimeToStr, 'oracle', default=True)(args)
    return exp.ToChar.from_arg_list(args)