import copy
import datetime
import re
from django.db import DatabaseError
from django.db.backends.base.schema import (
from django.utils.duration import duration_iso_string
def _is_identity_column(self, table_name, column_name):
    with self.connection.cursor() as cursor:
        cursor.execute("\n                SELECT\n                    CASE WHEN identity_column = 'YES' THEN 1 ELSE 0 END\n                FROM user_tab_cols\n                WHERE table_name = %s AND\n                      column_name = %s\n                ", [self.normalize_name(table_name), self.normalize_name(column_name)])
        row = cursor.fetchone()
        return row[0] if row else False