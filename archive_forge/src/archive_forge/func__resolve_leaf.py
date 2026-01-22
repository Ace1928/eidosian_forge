import operator
from functools import reduce
from django.core.exceptions import EmptyResultSet, FullResultSet
from django.db.models.expressions import Case, When
from django.db.models.functions import Mod
from django.db.models.lookups import Exact
from django.utils import tree
from django.utils.functional import cached_property
@staticmethod
def _resolve_leaf(expr, query, *args, **kwargs):
    if hasattr(expr, 'resolve_expression'):
        expr = expr.resolve_expression(query, *args, **kwargs)
    return expr