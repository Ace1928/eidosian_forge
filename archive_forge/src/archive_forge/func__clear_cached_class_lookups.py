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
@classmethod
def _clear_cached_class_lookups(cls):
    for subclass in subclasses(cls):
        subclass.get_class_lookups.cache_clear()