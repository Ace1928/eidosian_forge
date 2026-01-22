from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.tokens import TokenType
def _transform_create(expression: exp.Expression) -> exp.Expression:
    """Move primary key to a column and enforce auto_increment on primary keys."""
    schema = expression.this
    if isinstance(expression, exp.Create) and isinstance(schema, exp.Schema):
        defs = {}
        primary_key = None
        for e in schema.expressions:
            if isinstance(e, exp.ColumnDef):
                defs[e.name] = e
            elif isinstance(e, exp.PrimaryKey):
                primary_key = e
        if primary_key and len(primary_key.expressions) == 1:
            column = defs[primary_key.expressions[0].name]
            column.append('constraints', exp.ColumnConstraint(kind=exp.PrimaryKeyColumnConstraint()))
            schema.expressions.remove(primary_key)
        else:
            for column in defs.values():
                auto_increment = None
                for constraint in column.constraints:
                    if isinstance(constraint.kind, exp.PrimaryKeyColumnConstraint):
                        break
                    if isinstance(constraint.kind, exp.AutoIncrementColumnConstraint):
                        auto_increment = constraint
                if auto_increment:
                    column.constraints.remove(auto_increment)
    return expression