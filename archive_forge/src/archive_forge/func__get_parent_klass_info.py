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
def _get_parent_klass_info(klass_info):
    concrete_model = klass_info['model']._meta.concrete_model
    for parent_model, parent_link in concrete_model._meta.parents.items():
        parent_list = parent_model._meta.get_parent_list()
        yield {'model': parent_model, 'field': parent_link, 'reverse': False, 'select_fields': [select_index for select_index in klass_info['select_fields'] if self.select[select_index][0].target.model == parent_model or self.select[select_index][0].target.model in parent_list]}