from collections import namedtuple
import sqlparse
from MySQLdb.constants import FIELD_TYPE
from django.db.backends.base.introspection import BaseDatabaseIntrospection
from django.db.backends.base.introspection import FieldInfo as BaseFieldInfo
from django.db.backends.base.introspection import TableInfo as BaseTableInfo
from django.db.models import Index
from django.utils.datastructures import OrderedSet
def _parse_constraint_columns(self, check_clause, columns):
    check_columns = OrderedSet()
    statement = sqlparse.parse(check_clause)[0]
    tokens = (token for token in statement.flatten() if not token.is_whitespace)
    for token in tokens:
        if token.ttype == sqlparse.tokens.Name and self.connection.ops.quote_name(token.value) == token.value and (token.value[1:-1] in columns):
            check_columns.add(token.value[1:-1])
    return check_columns