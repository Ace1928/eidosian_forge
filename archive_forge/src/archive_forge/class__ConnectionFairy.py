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
class _ConnectionFairy(PoolProxiedConnection):
    """Proxies a DBAPI connection and provides return-on-dereference
    support.

    This is an internal object used by the :class:`_pool.Pool` implementation
    to provide context management to a DBAPI connection delivered by
    that :class:`_pool.Pool`.   The public facing interface for this class
    is described by the :class:`.PoolProxiedConnection` class.  See that
    class for public API details.

    The name "fairy" is inspired by the fact that the
    :class:`._ConnectionFairy` object's lifespan is transitory, as it lasts
    only for the length of a specific DBAPI connection being checked out from
    the pool, and additionally that as a transparent proxy, it is mostly
    invisible.

    .. seealso::

        :class:`.PoolProxiedConnection`

        :class:`.ConnectionPoolEntry`


    """
    __slots__ = ('dbapi_connection', '_connection_record', '_echo', '_pool', '_counter', '__weakref__', '__dict__')
    pool: Pool
    dbapi_connection: DBAPIConnection
    _echo: log._EchoFlagType

    def __init__(self, pool: Pool, dbapi_connection: DBAPIConnection, connection_record: _ConnectionRecord, echo: log._EchoFlagType):
        self._pool = pool
        self._counter = 0
        self.dbapi_connection = dbapi_connection
        self._connection_record = connection_record
        self._echo = echo
    _connection_record: Optional[_ConnectionRecord]

    @property
    def driver_connection(self) -> Optional[Any]:
        if self._connection_record is None:
            return None
        return self._connection_record.driver_connection

    @property
    @util.deprecated('2.0', "The _ConnectionFairy.connection attribute is deprecated; please use 'driver_connection'")
    def connection(self) -> DBAPIConnection:
        return self.dbapi_connection

    @classmethod
    def _checkout(cls, pool: Pool, threadconns: Optional[threading.local]=None, fairy: Optional[_ConnectionFairy]=None) -> _ConnectionFairy:
        if not fairy:
            fairy = _ConnectionRecord.checkout(pool)
            if threadconns is not None:
                threadconns.current = weakref.ref(fairy)
        assert fairy._connection_record is not None, "can't 'checkout' a detached connection fairy"
        assert fairy.dbapi_connection is not None, "can't 'checkout' an invalidated connection fairy"
        fairy._counter += 1
        if not pool.dispatch.checkout and (not pool._pre_ping) or fairy._counter != 1:
            return fairy
        attempts = 2
        while attempts > 0:
            connection_is_fresh = fairy._connection_record.fresh
            fairy._connection_record.fresh = False
            try:
                if pool._pre_ping:
                    if not connection_is_fresh:
                        if fairy._echo:
                            pool.logger.debug('Pool pre-ping on connection %s', fairy.dbapi_connection)
                        result = pool._dialect._do_ping_w_event(fairy.dbapi_connection)
                        if not result:
                            if fairy._echo:
                                pool.logger.debug('Pool pre-ping on connection %s failed, will invalidate pool', fairy.dbapi_connection)
                            raise exc.InvalidatePoolError()
                    elif fairy._echo:
                        pool.logger.debug('Connection %s is fresh, skipping pre-ping', fairy.dbapi_connection)
                pool.dispatch.checkout(fairy.dbapi_connection, fairy._connection_record, fairy)
                return fairy
            except exc.DisconnectionError as e:
                if e.invalidate_pool:
                    pool.logger.info('Disconnection detected on checkout, invalidating all pooled connections prior to current timestamp (reason: %r)', e)
                    fairy._connection_record.invalidate(e)
                    pool._invalidate(fairy, e, _checkin=False)
                else:
                    pool.logger.info('Disconnection detected on checkout, invalidating individual connection %s (reason: %r)', fairy.dbapi_connection, e)
                    fairy._connection_record.invalidate(e)
                try:
                    fairy.dbapi_connection = fairy._connection_record.get_connection()
                except BaseException as err:
                    with util.safe_reraise():
                        fairy._connection_record._checkin_failed(err, _fairy_was_created=True)
                        del fairy
                    raise
                attempts -= 1
            except BaseException as be_outer:
                with util.safe_reraise():
                    rec = fairy._connection_record
                    if rec is not None:
                        rec._checkin_failed(be_outer, _fairy_was_created=True)
                    del fairy
                raise
        pool.logger.info('Reconnection attempts exhausted on checkout')
        fairy.invalidate()
        raise exc.InvalidRequestError('This connection is closed')

    def _checkout_existing(self) -> _ConnectionFairy:
        return _ConnectionFairy._checkout(self._pool, fairy=self)

    def _checkin(self, transaction_was_reset: bool=False) -> None:
        _finalize_fairy(self.dbapi_connection, self._connection_record, self._pool, None, self._echo, transaction_was_reset=transaction_was_reset, fairy=self)

    def _close(self) -> None:
        self._checkin()

    def _reset(self, pool: Pool, transaction_was_reset: bool, terminate_only: bool, asyncio_safe: bool) -> None:
        if pool.dispatch.reset:
            pool.dispatch.reset(self.dbapi_connection, self._connection_record, PoolResetState(transaction_was_reset=transaction_was_reset, terminate_only=terminate_only, asyncio_safe=asyncio_safe))
        if not asyncio_safe:
            return
        if pool._reset_on_return is reset_rollback:
            if transaction_was_reset:
                if self._echo:
                    pool.logger.debug('Connection %s reset, transaction already reset', self.dbapi_connection)
            else:
                if self._echo:
                    pool.logger.debug('Connection %s rollback-on-return', self.dbapi_connection)
                pool._dialect.do_rollback(self)
        elif pool._reset_on_return is reset_commit:
            if self._echo:
                pool.logger.debug('Connection %s commit-on-return', self.dbapi_connection)
            pool._dialect.do_commit(self)

    @property
    def _logger(self) -> log._IdentifiedLoggerType:
        return self._pool.logger

    @property
    def is_valid(self) -> bool:
        return self.dbapi_connection is not None

    @property
    def is_detached(self) -> bool:
        return self._connection_record is None

    @util.ro_memoized_property
    def info(self) -> _InfoType:
        if self._connection_record is None:
            return {}
        else:
            return self._connection_record.info

    @util.ro_non_memoized_property
    def record_info(self) -> Optional[_InfoType]:
        if self._connection_record is None:
            return None
        else:
            return self._connection_record.record_info

    def invalidate(self, e: Optional[BaseException]=None, soft: bool=False) -> None:
        if self.dbapi_connection is None:
            util.warn("Can't invalidate an already-closed connection.")
            return
        if self._connection_record:
            self._connection_record.invalidate(e=e, soft=soft)
        if not soft:
            self.dbapi_connection = None
            self._checkin()

    def cursor(self, *args: Any, **kwargs: Any) -> DBAPICursor:
        assert self.dbapi_connection is not None
        return self.dbapi_connection.cursor(*args, **kwargs)

    def __getattr__(self, key: str) -> Any:
        return getattr(self.dbapi_connection, key)

    def detach(self) -> None:
        if self._connection_record is not None:
            rec = self._connection_record
            rec.fairy_ref = None
            rec.dbapi_connection = None
            self._pool._do_return_conn(self._connection_record)
            self.info = self.info.copy()
            self._connection_record = None
            if self._pool.dispatch.detach:
                self._pool.dispatch.detach(self.dbapi_connection, rec)

    def close(self) -> None:
        self._counter -= 1
        if self._counter == 0:
            self._checkin()

    def _close_special(self, transaction_reset: bool=False) -> None:
        self._counter -= 1
        if self._counter == 0:
            self._checkin(transaction_was_reset=transaction_reset)