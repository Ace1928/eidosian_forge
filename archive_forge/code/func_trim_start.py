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
def trim_start(self, names_with_path):
    """
        Trim joins from the start of the join path. The candidates for trim
        are the PathInfos in names_with_path structure that are m2m joins.

        Also set the select column so the start matches the join.

        This method is meant to be used for generating the subquery joins &
        cols in split_exclude().

        Return a lookup usable for doing outerq.filter(lookup=self) and a
        boolean indicating if the joins in the prefix contain a LEFT OUTER join.
        _"""
    all_paths = []
    for _, paths in names_with_path:
        all_paths.extend(paths)
    contains_louter = False
    lookup_tables = [t for t in self.alias_map if t in self._lookup_joins or t == self.base_table]
    for trimmed_paths, path in enumerate(all_paths):
        if path.m2m:
            break
        if self.alias_map[lookup_tables[trimmed_paths + 1]].join_type == LOUTER:
            contains_louter = True
        alias = lookup_tables[trimmed_paths]
        self.unref_alias(alias)
    join_field = path.join_field.field
    paths_in_prefix = trimmed_paths
    trimmed_prefix = []
    for name, path in names_with_path:
        if paths_in_prefix - len(path) < 0:
            break
        trimmed_prefix.append(name)
        paths_in_prefix -= len(path)
    trimmed_prefix.append(join_field.foreign_related_fields[0].name)
    trimmed_prefix = LOOKUP_SEP.join(trimmed_prefix)
    first_join = self.alias_map[lookup_tables[trimmed_paths + 1]]
    if first_join.join_type != LOUTER and (not first_join.filtered_relation):
        select_fields = [r[0] for r in join_field.related_fields]
        select_alias = lookup_tables[trimmed_paths + 1]
        self.unref_alias(lookup_tables[trimmed_paths])
        extra_restriction = join_field.get_extra_restriction(None, lookup_tables[trimmed_paths + 1])
        if extra_restriction:
            self.where.add(extra_restriction, AND)
    else:
        select_fields = [r[1] for r in join_field.related_fields]
        select_alias = lookup_tables[trimmed_paths]
    for table in self.alias_map:
        if self.alias_refcount[table] > 0:
            self.alias_map[table] = self.base_table_class(self.alias_map[table].table_name, table)
            break
    self.set_select([f.get_col(select_alias) for f in select_fields])
    return (trimmed_prefix, contains_louter)