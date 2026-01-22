import functools
import inspect
import logging
from collections import namedtuple
from django.core.exceptions import FieldError
from django.db import DEFAULT_DB_ALIAS, DatabaseError, connections
from django.db.models.constants import LOOKUP_SEP
from django.utils import tree
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
class FilteredRelation:
    """Specify custom filtering in the ON clause of SQL joins."""

    def __init__(self, relation_name, *, condition=Q()):
        if not relation_name:
            raise ValueError('relation_name cannot be empty.')
        self.relation_name = relation_name
        self.alias = None
        if not isinstance(condition, Q):
            raise ValueError('condition argument must be a Q() instance.')
        self.condition = condition
        self.resolved_condition = None

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.relation_name == other.relation_name and self.alias == other.alias and (self.condition == other.condition)

    def clone(self):
        clone = FilteredRelation(self.relation_name, condition=self.condition)
        clone.alias = self.alias
        if (resolved_condition := self.resolved_condition) is not None:
            clone.resolved_condition = resolved_condition.clone()
        return clone

    def relabeled_clone(self, change_map):
        clone = self.clone()
        if (resolved_condition := clone.resolved_condition):
            clone.resolved_condition = resolved_condition.relabeled_clone(change_map)
        return clone

    def resolve_expression(self, query, reuse, *args, **kwargs):
        clone = self.clone()
        clone.resolved_condition = query.build_filter(self.condition, can_reuse=reuse, allow_joins=True, split_subq=False, update_join_types=False)[0]
        return clone

    def as_sql(self, compiler, connection):
        return compiler.compile(self.resolved_condition)