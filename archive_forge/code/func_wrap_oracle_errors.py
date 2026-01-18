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
@contextmanager
def wrap_oracle_errors():
    try:
        yield
    except Database.DatabaseError as e:
        x = e.args[0]
        if hasattr(x, 'code') and hasattr(x, 'message') and (x.code == 2091) and ('ORA-02291' in x.message or 'ORA-00001' in x.message):
            raise IntegrityError(*tuple(e.args))
        raise