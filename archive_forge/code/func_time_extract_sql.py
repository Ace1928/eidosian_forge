import json
from functools import lru_cache, partial
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.postgresql.psycopg_any import (
from django.db.backends.utils import split_tzname_delta
from django.db.models.constants import OnConflict
from django.db.models.functions import Cast
from django.utils.regex_helper import _lazy_re_compile
def time_extract_sql(self, lookup_type, sql, params):
    if lookup_type == 'second':
        return (f'EXTRACT(SECOND FROM DATE_TRUNC(%s, {sql}))', ('second', *params))
    return self.date_extract_sql(lookup_type, sql, params)