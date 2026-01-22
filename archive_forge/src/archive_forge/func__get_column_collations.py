from collections import namedtuple
import sqlparse
from django.db import DatabaseError
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo
from django.db.models import Index
from django.utils.regex_helper import _lazy_re_compile
def _get_column_collations(self, cursor, table_name):
    row = cursor.execute("\n            SELECT sql\n            FROM sqlite_master\n            WHERE type = 'table' AND name = %s\n        ", [table_name]).fetchone()
    if not row:
        return {}
    sql = row[0]
    columns = str(sqlparse.parse(sql)[0][-1]).strip('()').split(', ')
    collations = {}
    for column in columns:
        tokens = column[1:].split()
        column_name = tokens[0].strip('"')
        for index, token in enumerate(tokens):
            if token == 'COLLATE':
                collation = tokens[index + 1]
                break
        else:
            collation = None
        collations[column_name] = collation
    return collations