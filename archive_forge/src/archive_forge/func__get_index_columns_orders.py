from collections import namedtuple
import sqlparse
from django.db import DatabaseError
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo
from django.db.models import Index
from django.utils.regex_helper import _lazy_re_compile
def _get_index_columns_orders(self, sql):
    tokens = sqlparse.parse(sql)[0]
    for token in tokens:
        if isinstance(token, sqlparse.sql.Parenthesis):
            columns = str(token).strip('()').split(', ')
            return ['DESC' if info.endswith('DESC') else 'ASC' for info in columns]
    return None