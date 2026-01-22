import datetime
import decimal
import uuid
from functools import lru_cache
from itertools import chain
from django.conf import settings
from django.core.exceptions import FieldError
from django.db import DatabaseError, NotSupportedError, models
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.constants import OnConflict
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.functional import cached_property
def __references_graph(self, table_name):
    query = '\n        WITH tables AS (\n            SELECT %s name\n            UNION\n            SELECT sqlite_master.name\n            FROM sqlite_master\n            JOIN tables ON (sql REGEXP %s || tables.name || %s)\n        ) SELECT name FROM tables;\n        '
    params = (table_name, '(?i)\\s+references\\s+("|\\\')?', '("|\\\')?\\s*\\(')
    with self.connection.cursor() as cursor:
        results = cursor.execute(query, params)
        return [row[0] for row in results.fetchall()]