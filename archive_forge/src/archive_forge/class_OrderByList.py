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
class OrderByList(Func):
    allowed_default = False
    template = 'ORDER BY %(expressions)s'

    def __init__(self, *expressions, **extra):
        expressions = (OrderBy(F(expr[1:]), descending=True) if isinstance(expr, str) and expr[0] == '-' else expr for expr in expressions)
        super().__init__(*expressions, **extra)

    def as_sql(self, *args, **kwargs):
        if not self.source_expressions:
            return ('', ())
        return super().as_sql(*args, **kwargs)

    def get_group_by_cols(self):
        group_by_cols = []
        for order_by in self.get_source_expressions():
            group_by_cols.extend(order_by.get_group_by_cols())
        return group_by_cols