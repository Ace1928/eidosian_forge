import datetime
import decimal
import warnings
from collections.abc import Mapping
from itertools import chain, tee
from sqlite3 import dbapi2 as Database
from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError
from django.db.backends.base.base import BaseDatabaseWrapper
from django.utils.asyncio import async_unsafe
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.regex_helper import _lazy_re_compile
from ._functions import register as register_functions
from .client import DatabaseClient
from .creation import DatabaseCreation
from .features import DatabaseFeatures
from .introspection import DatabaseIntrospection
from .operations import DatabaseOperations
from .schema import DatabaseSchemaEditor
class SQLiteCursorWrapper(Database.Cursor):
    """
    Django uses the "format" and "pyformat" styles, but Python's sqlite3 module
    supports neither of these styles.

    This wrapper performs the following conversions:

    - "format" style to "qmark" style
    - "pyformat" style to "named" style

    In both cases, if you want to use a literal "%s", you'll need to use "%%s".
    """

    def execute(self, query, params=None):
        if params is None:
            return super().execute(query)
        param_names = list(params) if isinstance(params, Mapping) else None
        query = self.convert_query(query, param_names=param_names)
        return super().execute(query, params)

    def executemany(self, query, param_list):
        peekable, param_list = tee(iter(param_list))
        if (params := next(peekable, None)) and isinstance(params, Mapping):
            param_names = list(params)
        else:
            param_names = None
        query = self.convert_query(query, param_names=param_names)
        return super().executemany(query, param_list)

    def convert_query(self, query, *, param_names=None):
        if param_names is None:
            return FORMAT_QMARK_REGEX.sub('?', query).replace('%%', '%')
        else:
            return query % {name: f':{name}' for name in param_names}