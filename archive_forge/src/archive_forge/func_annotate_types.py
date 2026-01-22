from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def annotate_types(expression: E, schema: t.Optional[t.Dict | Schema]=None, annotators: t.Optional[t.Dict[t.Type[E], t.Callable[[TypeAnnotator, E], E]]]=None, coerces_to: t.Optional[t.Dict[exp.DataType.Type, t.Set[exp.DataType.Type]]]=None) -> E:
    """
    Infers the types of an expression, annotating its AST accordingly.

    Example:
        >>> import sqlglot
        >>> schema = {"y": {"cola": "SMALLINT"}}
        >>> sql = "SELECT x.cola + 2.5 AS cola FROM (SELECT y.cola AS cola FROM y AS y) AS x"
        >>> annotated_expr = annotate_types(sqlglot.parse_one(sql), schema=schema)
        >>> annotated_expr.expressions[0].type.this  # Get the type of "x.cola + 2.5 AS cola"
        <Type.DOUBLE: 'DOUBLE'>

    Args:
        expression: Expression to annotate.
        schema: Database schema.
        annotators: Maps expression type to corresponding annotation function.
        coerces_to: Maps expression type to set of types that it can be coerced into.

    Returns:
        The expression annotated with types.
    """
    schema = ensure_schema(schema)
    return TypeAnnotator(schema, annotators, coerces_to).annotate(expression)