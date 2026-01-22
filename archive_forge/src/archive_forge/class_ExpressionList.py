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
class ExpressionList(Func):
    """
    An expression containing multiple expressions. Can be used to provide a
    list of expressions as an argument to another expression, like a partition
    clause.
    """
    template = '%(expressions)s'

    def __init__(self, *expressions, **extra):
        if not expressions:
            raise ValueError('%s requires at least one expression.' % self.__class__.__name__)
        super().__init__(*expressions, **extra)

    def __str__(self):
        return self.arg_joiner.join((str(arg) for arg in self.source_expressions))

    def as_sqlite(self, compiler, connection, **extra_context):
        return self.as_sql(compiler, connection, **extra_context)

    def get_group_by_cols(self):
        group_by_cols = []
        for partition in self.get_source_expressions():
            group_by_cols.extend(partition.get_group_by_cols())
        return group_by_cols