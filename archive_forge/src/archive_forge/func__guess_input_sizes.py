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
def _guess_input_sizes(self, params_list):
    if hasattr(params_list[0], 'keys'):
        sizes = {}
        for params in params_list:
            for k, value in params.items():
                if value.input_size:
                    sizes[k] = value.input_size
        if sizes:
            self.setinputsizes(**sizes)
    else:
        sizes = [None] * len(params_list[0])
        for params in params_list:
            for i, value in enumerate(params):
                if value.input_size:
                    sizes[i] = value.input_size
        if sizes:
            self.setinputsizes(*sizes)