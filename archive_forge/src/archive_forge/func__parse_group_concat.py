from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_group_concat(self) -> t.Optional[exp.Expression]:

    def concat_exprs(node: t.Optional[exp.Expression], exprs: t.List[exp.Expression]) -> exp.Expression:
        if isinstance(node, exp.Distinct) and len(node.expressions) > 1:
            concat_exprs = [self.expression(exp.Concat, expressions=node.expressions, safe=True)]
            node.set('expressions', concat_exprs)
            return node
        if len(exprs) == 1:
            return exprs[0]
        return self.expression(exp.Concat, expressions=args, safe=True)
    args = self._parse_csv(self._parse_lambda)
    if args:
        order = args[-1] if isinstance(args[-1], exp.Order) else None
        if order:
            args[-1] = order.this
            order.set('this', concat_exprs(order.this, args))
        this = order or concat_exprs(args[0], args)
    else:
        this = None
    separator = self._parse_field() if self._match(TokenType.SEPARATOR) else None
    return self.expression(exp.GroupConcat, this=this, separator=separator)