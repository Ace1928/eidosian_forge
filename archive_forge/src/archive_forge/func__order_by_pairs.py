import collections
import json
import re
from functools import partial
from itertools import chain
from django.core.exceptions import EmptyResultSet, FieldError, FullResultSet
from django.db import DatabaseError, NotSupportedError
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import F, OrderBy, RawSQL, Ref, Value
from django.db.models.functions import Cast, Random
from django.db.models.lookups import Lookup
from django.db.models.query_utils import select_related_descend
from django.db.models.sql.constants import (
from django.db.models.sql.query import Query, get_order_dir
from django.db.models.sql.where import AND
from django.db.transaction import TransactionManagementError
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from django.utils.regex_helper import _lazy_re_compile
def _order_by_pairs(self):
    if self.query.extra_order_by:
        ordering = self.query.extra_order_by
    elif not self.query.default_ordering:
        ordering = self.query.order_by
    elif self.query.order_by:
        ordering = self.query.order_by
    elif (meta := self.query.get_meta()) and meta.ordering:
        ordering = meta.ordering
        self._meta_ordering = ordering
    else:
        ordering = []
    if self.query.standard_ordering:
        default_order, _ = ORDER_DIR['ASC']
    else:
        default_order, _ = ORDER_DIR['DESC']
    selected_exprs = {}
    if ordering and (select := self.select):
        for ordinal, (expr, _, alias) in enumerate(select, start=1):
            pos_expr = PositionRef(ordinal, alias, expr)
            if alias:
                selected_exprs[alias] = pos_expr
            selected_exprs[expr] = pos_expr
    for field in ordering:
        if hasattr(field, 'resolve_expression'):
            if isinstance(field, Value):
                field = Cast(field, field.output_field)
            if not isinstance(field, OrderBy):
                field = field.asc()
            if not self.query.standard_ordering:
                field = field.copy()
                field.reverse_ordering()
            select_ref = selected_exprs.get(field.expression)
            if select_ref or (isinstance(field.expression, F) and (select_ref := selected_exprs.get(field.expression.name))):
                if field.nulls_first is None and field.nulls_last is None or self.connection.features.supports_order_by_nulls_modifier:
                    field = field.copy()
                    field.expression = select_ref
                elif self.query.combinator:
                    field = field.copy()
                    field.expression = Ref(select_ref.refs, select_ref.source)
            yield (field, select_ref is not None)
            continue
        if field == '?':
            yield (OrderBy(Random()), False)
            continue
        col, order = get_order_dir(field, default_order)
        descending = order == 'DESC'
        if (select_ref := selected_exprs.get(col)):
            yield (OrderBy(select_ref, descending=descending), True)
            continue
        if col in self.query.annotations:
            if self.query.combinator and self.select:
                expr = F(col)
            else:
                expr = self.query.annotations[col]
                if isinstance(expr, Value):
                    expr = Cast(expr, expr.output_field)
            yield (OrderBy(expr, descending=descending), False)
            continue
        if '.' in field:
            table, col = col.split('.', 1)
            yield (OrderBy(RawSQL('%s.%s' % (self.quote_name_unless_alias(table), col), []), descending=descending), False)
            continue
        if self.query.extra and col in self.query.extra:
            if col in self.query.extra_select:
                yield (OrderBy(Ref(col, RawSQL(*self.query.extra[col])), descending=descending), True)
            else:
                yield (OrderBy(RawSQL(*self.query.extra[col]), descending=descending), False)
        elif self.query.combinator and self.select:
            yield (OrderBy(F(col), descending=descending), False)
        else:
            yield from self.find_ordering_name(field, self.query.get_meta(), default_order=default_order)