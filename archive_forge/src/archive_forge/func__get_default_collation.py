import copy
import datetime
import re
from django.db import DatabaseError
from django.db.backends.base.schema import (
from django.utils.duration import duration_iso_string
def _get_default_collation(self, table_name):
    with self.connection.cursor() as cursor:
        cursor.execute('\n                SELECT default_collation FROM user_tables WHERE table_name = %s\n                ', [self.normalize_name(table_name)])
        return cursor.fetchone()[0]