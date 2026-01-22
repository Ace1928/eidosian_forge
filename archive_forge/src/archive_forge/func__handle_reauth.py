from __future__ import annotations
import sys
import traceback
from collections import abc
from typing import (
from bson.son import SON
from pymongo import ASCENDING
from pymongo.errors import (
from pymongo.hello import HelloCompat
def _handle_reauth(func: F) -> F:

    def inner(*args: Any, **kwargs: Any) -> Any:
        no_reauth = kwargs.pop('no_reauth', False)
        from pymongo.message import _BulkWriteContext
        from pymongo.pool import Connection
        try:
            return func(*args, **kwargs)
        except OperationFailure as exc:
            if no_reauth:
                raise
            if exc.code == _REAUTHENTICATION_REQUIRED_CODE:
                conn = None
                for arg in args:
                    if isinstance(arg, Connection):
                        conn = arg
                        break
                    if isinstance(arg, _BulkWriteContext):
                        conn = arg.conn
                        break
                if conn:
                    conn.authenticate(reauthenticate=True)
                else:
                    raise
                return func(*args, **kwargs)
            raise
    return cast(F, inner)