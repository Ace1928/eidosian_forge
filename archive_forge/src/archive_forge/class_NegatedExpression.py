import copy
import datetime
import functools
import inspect
from collections import defaultdict
from decimal import Decimal
from types import NoneType
from uuid import UUID
from django.core.exceptions import EmptyResultSet, FieldError, FullResultSet
from django.db import DatabaseError, NotSupportedError, connection
from django.db.models import fields
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import Q
from django.utils.deconstruct import deconstructible
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
class NegatedExpression(ExpressionWrapper):
    """The logical negation of a conditional expression."""

    def __init__(self, expression):
        super().__init__(expression, output_field=fields.BooleanField())

    def __invert__(self):
        return self.expression.copy()

    def as_sql(self, compiler, connection):
        try:
            sql, params = super().as_sql(compiler, connection)
        except EmptyResultSet:
            features = compiler.connection.features
            if not features.supports_boolean_expr_in_select_clause:
                return ('1=1', ())
            return compiler.compile(Value(True))
        ops = compiler.connection.ops
        if not ops.conditional_expression_supported_in_where_clause(self.expression):
            return (f'CASE WHEN {sql} = 0 THEN 1 ELSE 0 END', params)
        return (f'NOT {sql}', params)

    def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
        resolved = super().resolve_expression(query, allow_joins, reuse, summarize, for_save)
        if not getattr(resolved.expression, 'conditional', False):
            raise TypeError('Cannot negate non-conditional expressions.')
        return resolved

    def select_format(self, compiler, sql, params):
        expression_supported_in_where_clause = compiler.connection.ops.conditional_expression_supported_in_where_clause
        if not compiler.connection.features.supports_boolean_expr_in_select_clause and expression_supported_in_where_clause(self.expression):
            sql = 'CASE WHEN {} THEN 1 ELSE 0 END'.format(sql)
        return (sql, params)