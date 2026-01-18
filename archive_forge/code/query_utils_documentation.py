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

        Remove given lookup from instance lookups. For use in tests only as
        it's not thread-safe.
        