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
def trim_joins(self, targets, joins, path):
    """
        The 'target' parameter is the final field being joined to, 'joins'
        is the full list of join aliases. The 'path' contain the PathInfos
        used to create the joins.

        Return the final target field and table alias and the new active
        joins.

        Always trim any direct join if the target column is already in the
        previous table. Can't trim reverse joins as it's unknown if there's
        anything on the other side of the join.
        """
    joins = joins[:]
    for pos, info in enumerate(reversed(path)):
        if len(joins) == 1 or not info.direct:
            break
        if info.filtered_relation:
            break
        join_targets = {t.column for t in info.join_field.foreign_related_fields}
        cur_targets = {t.column for t in targets}
        if not cur_targets.issubset(join_targets):
            break
        targets_dict = {r[1].column: r[0] for r in info.join_field.related_fields if r[1].column in cur_targets}
        targets = tuple((targets_dict[t.column] for t in targets))
        self.unref_alias(joins.pop())
    return (targets, joins[-1], joins)