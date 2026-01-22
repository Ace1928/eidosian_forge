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
def _get_defer_select_mask(self, opts, mask, select_mask=None):
    if select_mask is None:
        select_mask = {}
    select_mask[opts.pk] = {}
    for field in opts.concrete_fields:
        field_mask = mask.pop(field.name, None)
        field_att_mask = mask.pop(field.attname, None)
        if field_mask is None and field_att_mask is None:
            select_mask.setdefault(field, {})
        elif field_mask:
            if not field.is_relation:
                raise FieldError(next(iter(field_mask)))
            field_select_mask = select_mask.setdefault(field, {})
            related_model = field.remote_field.model._meta.concrete_model
            self._get_defer_select_mask(related_model._meta, field_mask, field_select_mask)
    for field_name, field_mask in mask.items():
        if (filtered_relation := self._filtered_relations.get(field_name)):
            relation = opts.get_field(filtered_relation.relation_name)
            field_select_mask = select_mask.setdefault((field_name, relation), {})
            field = relation.field
        else:
            reverse_rel = opts.get_field(field_name)
            if not hasattr(reverse_rel, 'field'):
                continue
            field = reverse_rel.field
            field_select_mask = select_mask.setdefault(field, {})
        related_model = field.model._meta.concrete_model
        self._get_defer_select_mask(related_model._meta, field_mask, field_select_mask)
    return select_mask