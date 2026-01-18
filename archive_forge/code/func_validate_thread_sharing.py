import _thread
import copy
import datetime
import logging
import threading
import time
import warnings
import zoneinfo
from collections import deque
from contextlib import contextmanager
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import DEFAULT_DB_ALIAS, DatabaseError, NotSupportedError
from django.db.backends import utils
from django.db.backends.base.validation import BaseDatabaseValidation
from django.db.backends.signals import connection_created
from django.db.backends.utils import debug_transaction
from django.db.transaction import TransactionManagementError
from django.db.utils import DatabaseErrorWrapper
from django.utils.asyncio import async_unsafe
from django.utils.functional import cached_property
def validate_thread_sharing(self):
    """
        Validate that the connection isn't accessed by another thread than the
        one which originally created it, unless the connection was explicitly
        authorized to be shared between threads (via the `inc_thread_sharing()`
        method). Raise an exception if the validation fails.
        """
    if not (self.allow_thread_sharing or self._thread_ident == _thread.get_ident()):
        raise DatabaseError("DatabaseWrapper objects created in a thread can only be used in that same thread. The object with alias '%s' was created in thread id %s and this is thread id %s." % (self.alias, self._thread_ident, _thread.get_ident()))