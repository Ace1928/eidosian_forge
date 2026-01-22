import copy
import difflib
import functools
import sys
from collections import Counter, namedtuple
from collections.abc import Iterator, Mapping
from itertools import chain, count, product
from string import ascii_uppercase
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db import DEFAULT_DB_ALIAS, NotSupportedError, connections
from django.db.models.aggregates import Count
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import (
from django.db.models.fields import Field
from django.db.models.fields.related_lookups import MultiColSource
from django.db.models.lookups import Lookup
from django.db.models.query_utils import (
from django.db.models.sql.constants import INNER, LOUTER, ORDER_DIR, SINGLE
from django.db.models.sql.datastructures import BaseTable, Empty, Join, MultiJoin
from django.db.models.sql.where import AND, OR, ExtraWhere, NothingNode, WhereNode
from django.utils.functional import cached_property
from django.utils.regex_helper import _lazy_re_compile
from django.utils.tree import Node
def add_ordering(self, *ordering):
    """
        Add items from the 'ordering' sequence to the query's "order by"
        clause. These items are either field names (not column names) --
        possibly with a direction prefix ('-' or '?') -- or OrderBy
        expressions.

        If 'ordering' is empty, clear all ordering from the query.
        """
    errors = []
    for item in ordering:
        if isinstance(item, str):
            if item == '?':
                continue
            item = item.removeprefix('-')
            if item in self.annotations:
                continue
            if self.extra and item in self.extra:
                continue
            self.names_to_path(item.split(LOOKUP_SEP), self.model._meta)
        elif not hasattr(item, 'resolve_expression'):
            errors.append(item)
        if getattr(item, 'contains_aggregate', False):
            raise FieldError('Using an aggregate in order_by() without also including it in annotate() is not allowed: %s' % item)
    if errors:
        raise FieldError('Invalid order_by arguments: %s' % errors)
    if ordering:
        self.order_by += ordering
    else:
        self.default_ordering = False