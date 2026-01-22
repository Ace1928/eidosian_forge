import collections
import json
import re
from functools import partial
from itertools import chain
from django.core.exceptions import EmptyResultSet, FieldError, FullResultSet
from django.db import DatabaseError, NotSupportedError
from django.db.models.constants import LOOKUP_SEP
from django.db.models.expressions import F, OrderBy, RawSQL, Ref, Value
from django.db.models.functions import Cast, Random
from django.db.models.lookups import Lookup
from django.db.models.query_utils import select_related_descend
from django.db.models.sql.constants import (
from django.db.models.sql.query import Query, get_order_dir
from django.db.models.sql.where import AND
from django.db.transaction import TransactionManagementError
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from django.utils.regex_helper import _lazy_re_compile
def _get_field_choices():
    """Yield all allowed field paths in breadth-first search order."""
    queue = collections.deque([(None, self.klass_info)])
    while queue:
        parent_path, klass_info = queue.popleft()
        if parent_path is None:
            path = []
            yield 'self'
        else:
            field = klass_info['field']
            if klass_info['reverse']:
                field = field.remote_field
            path = parent_path + [field.name]
            yield LOOKUP_SEP.join(path)
        queue.extend(((path, klass_info) for klass_info in _get_parent_klass_info(klass_info)))
        queue.extend(((path, klass_info) for klass_info in klass_info.get('related_klass_infos', [])))