from __future__ import annotations
from collections import deque
import dataclasses
from enum import Enum
import threading
import time
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
def _finalize_fairy(dbapi_connection: Optional[DBAPIConnection], connection_record: Optional[_ConnectionRecord], pool: Pool, ref: Optional[weakref.ref[_ConnectionFairy]], echo: Optional[log._EchoFlagType], transaction_was_reset: bool=False, fairy: Optional[_ConnectionFairy]=None) -> None:
    """Cleanup for a :class:`._ConnectionFairy` whether or not it's already
    been garbage collected.

    When using an async dialect no IO can happen here (without using
    a dedicated thread), since this is called outside the greenlet
    context and with an already running loop. In this case function
    will only log a message and raise a warning.
    """
    is_gc_cleanup = ref is not None
    if is_gc_cleanup:
        assert ref is not None
        _strong_ref_connection_records.pop(ref, None)
        assert connection_record is not None
        if connection_record.fairy_ref is not ref:
            return
        assert dbapi_connection is None
        dbapi_connection = connection_record.dbapi_connection
    elif fairy:
        _strong_ref_connection_records.pop(weakref.ref(fairy), None)
    dont_restore_gced = pool._dialect.is_async
    if dont_restore_gced:
        detach = connection_record is None or is_gc_cleanup
        can_manipulate_connection = not is_gc_cleanup
        can_close_or_terminate_connection = not pool._dialect.is_async or pool._dialect.has_terminate
        requires_terminate_for_close = pool._dialect.is_async and pool._dialect.has_terminate
    else:
        detach = connection_record is None
        can_manipulate_connection = can_close_or_terminate_connection = True
        requires_terminate_for_close = False
    if dbapi_connection is not None:
        if connection_record and echo:
            pool.logger.debug('Connection %r being returned to pool', dbapi_connection)
        try:
            if not fairy:
                assert connection_record is not None
                fairy = _ConnectionFairy(pool, dbapi_connection, connection_record, echo)
            assert fairy.dbapi_connection is dbapi_connection
            fairy._reset(pool, transaction_was_reset=transaction_was_reset, terminate_only=detach, asyncio_safe=can_manipulate_connection)
            if detach:
                if connection_record:
                    fairy._pool = pool
                    fairy.detach()
                if can_close_or_terminate_connection:
                    if pool.dispatch.close_detached:
                        pool.dispatch.close_detached(dbapi_connection)
                    pool._close_connection(dbapi_connection, terminate=requires_terminate_for_close)
        except BaseException as e:
            pool.logger.error('Exception during reset or similar', exc_info=True)
            if connection_record:
                connection_record.invalidate(e=e)
            if not isinstance(e, Exception):
                raise
        finally:
            if detach and is_gc_cleanup and dont_restore_gced:
                message = f'The garbage collector is trying to clean up non-checked-in connection {dbapi_connection!r}, which will be {('dropped, as it cannot be safely terminated' if not can_close_or_terminate_connection else 'terminated')}.  Please ensure that SQLAlchemy pooled connections are returned to the pool explicitly, either by calling ``close()`` or by using appropriate context managers to manage their lifecycle.'
                pool.logger.error(message)
                util.warn(message)
    if connection_record and connection_record.fairy_ref is not None:
        connection_record.checkin()
    if fairy is not None:
        fairy.dbapi_connection = None
        fairy._connection_record = None
    del dbapi_connection
    del connection_record
    del fairy