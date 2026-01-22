import datetime
import decimal
import os
import platform
from contextlib import contextmanager
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import IntegrityError
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.backends.oracle.oracledb_any import oracledb as Database
from django.db.backends.utils import debug_transaction
from django.utils.asyncio import async_unsafe
from django.utils.encoding import force_bytes, force_str
from django.utils.functional import cached_property
from .client import DatabaseClient  # NOQA
from .creation import DatabaseCreation  # NOQA
from .features import DatabaseFeatures  # NOQA
from .introspection import DatabaseIntrospection  # NOQA
from .operations import DatabaseOperations  # NOQA
from .schema import DatabaseSchemaEditor  # NOQA
from .utils import Oracle_datetime, dsn  # NOQA
from .validation import DatabaseValidation  # NOQA
class OracleParam:
    """
    Wrapper object for formatting parameters for Oracle. If the string
    representation of the value is large enough (greater than 4000 characters)
    the input size needs to be set as CLOB. Alternatively, if the parameter
    has an `input_size` attribute, then the value of the `input_size` attribute
    will be used instead. Otherwise, no input size will be set for the
    parameter when executing the query.
    """

    def __init__(self, param, cursor, strings_only=False):
        if settings.USE_TZ and (isinstance(param, datetime.datetime) and (not isinstance(param, Oracle_datetime))):
            param = Oracle_datetime.from_datetime(param)
        string_size = 0
        has_boolean_data_type = cursor.database.features.supports_boolean_expr_in_select_clause
        if not has_boolean_data_type:
            if param is True:
                param = 1
            elif param is False:
                param = 0
        if hasattr(param, 'bind_parameter'):
            self.force_bytes = param.bind_parameter(cursor)
        elif isinstance(param, (Database.Binary, datetime.timedelta)):
            self.force_bytes = param
        else:
            self.force_bytes = force_str(param, cursor.charset, strings_only)
            if isinstance(self.force_bytes, str):
                string_size = len(force_bytes(param, cursor.charset, strings_only))
        if hasattr(param, 'input_size'):
            self.input_size = param.input_size
        elif string_size > 4000:
            self.input_size = Database.CLOB
        elif isinstance(param, datetime.datetime):
            self.input_size = Database.TIMESTAMP
        elif has_boolean_data_type and isinstance(param, bool):
            self.input_size = Database.DB_TYPE_BOOLEAN
        else:
            self.input_size = None