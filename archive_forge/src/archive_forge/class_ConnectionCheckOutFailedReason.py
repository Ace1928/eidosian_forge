from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class ConnectionCheckOutFailedReason:
    """An enum that defines values for `reason` on a
    :class:`ConnectionCheckOutFailedEvent`.

    .. versionadded:: 3.9
    """
    TIMEOUT = 'timeout'
    'The connection check out attempt exceeded the specified timeout.'
    POOL_CLOSED = 'poolClosed'
    'The pool was previously closed, and cannot provide new connections.'
    CONN_ERROR = 'connectionError'
    'The connection check out attempt experienced an error while setting up\n    a new connection.\n    '