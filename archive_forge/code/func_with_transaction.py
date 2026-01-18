from the same snapshot timestamp. The server chooses the latest
from __future__ import annotations
import collections
import time
import uuid
from collections.abc import Mapping as _Mapping
from typing import (
from bson.binary import Binary
from bson.int64 import Int64
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot
from pymongo.cursor import _ConnectionManager
from pymongo.errors import (
from pymongo.helpers import _RETRYABLE_ERROR_CODES
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_type import SERVER_TYPE
from pymongo.write_concern import WriteConcern
def with_transaction(self, callback: Callable[[ClientSession], _T], read_concern: Optional[ReadConcern]=None, write_concern: Optional[WriteConcern]=None, read_preference: Optional[_ServerMode]=None, max_commit_time_ms: Optional[int]=None) -> _T:
    """Execute a callback in a transaction.

        This method starts a transaction on this session, executes ``callback``
        once, and then commits the transaction. For example::

          def callback(session):
              orders = session.client.db.orders
              inventory = session.client.db.inventory
              orders.insert_one({"sku": "abc123", "qty": 100}, session=session)
              inventory.update_one({"sku": "abc123", "qty": {"$gte": 100}},
                                   {"$inc": {"qty": -100}}, session=session)

          with client.start_session() as session:
              session.with_transaction(callback)

        To pass arbitrary arguments to the ``callback``, wrap your callable
        with a ``lambda`` like this::

          def callback(session, custom_arg, custom_kwarg=None):
              # Transaction operations...

          with client.start_session() as session:
              session.with_transaction(
                  lambda s: callback(s, "custom_arg", custom_kwarg=1))

        In the event of an exception, ``with_transaction`` may retry the commit
        or the entire transaction, therefore ``callback`` may be invoked
        multiple times by a single call to ``with_transaction``. Developers
        should be mindful of this possibility when writing a ``callback`` that
        modifies application state or has any other side-effects.
        Note that even when the ``callback`` is invoked multiple times,
        ``with_transaction`` ensures that the transaction will be committed
        at-most-once on the server.

        The ``callback`` should not attempt to start new transactions, but
        should simply run operations meant to be contained within a
        transaction. The ``callback`` should also not commit the transaction;
        this is handled automatically by ``with_transaction``. If the
        ``callback`` does commit or abort the transaction without error,
        however, ``with_transaction`` will return without taking further
        action.

        :class:`ClientSession` instances are **not thread-safe or fork-safe**.
        Consequently, the ``callback`` must not attempt to execute multiple
        operations concurrently.

        When ``callback`` raises an exception, ``with_transaction``
        automatically aborts the current transaction. When ``callback`` or
        :meth:`~ClientSession.commit_transaction` raises an exception that
        includes the ``"TransientTransactionError"`` error label,
        ``with_transaction`` starts a new transaction and re-executes
        the ``callback``.

        When :meth:`~ClientSession.commit_transaction` raises an exception with
        the ``"UnknownTransactionCommitResult"`` error label,
        ``with_transaction`` retries the commit until the result of the
        transaction is known.

        This method will cease retrying after 120 seconds has elapsed. This
        timeout is not configurable and any exception raised by the
        ``callback`` or by :meth:`ClientSession.commit_transaction` after the
        timeout is reached will be re-raised. Applications that desire a
        different timeout duration should not use this method.

        :Parameters:
          - `callback`: The callable ``callback`` to run inside a transaction.
            The callable must accept a single argument, this session. Note,
            under certain error conditions the callback may be run multiple
            times.
          - `read_concern` (optional): The
            :class:`~pymongo.read_concern.ReadConcern` to use for this
            transaction.
          - `write_concern` (optional): The
            :class:`~pymongo.write_concern.WriteConcern` to use for this
            transaction.
          - `read_preference` (optional): The read preference to use for this
            transaction. If ``None`` (the default) the :attr:`read_preference`
            of this :class:`Database` is used. See
            :mod:`~pymongo.read_preferences` for options.

        :Returns:
          The return value of the ``callback``.

        .. versionadded:: 3.9
        """
    start_time = time.monotonic()
    while True:
        self.start_transaction(read_concern, write_concern, read_preference, max_commit_time_ms)
        try:
            ret = callback(self)
        except Exception as exc:
            if self.in_transaction:
                self.abort_transaction()
            if isinstance(exc, PyMongoError) and exc.has_error_label('TransientTransactionError') and _within_time_limit(start_time):
                continue
            raise
        if not self.in_transaction:
            return ret
        while True:
            try:
                self.commit_transaction()
            except PyMongoError as exc:
                if exc.has_error_label('UnknownTransactionCommitResult') and _within_time_limit(start_time) and (not _max_time_expired_error(exc)):
                    continue
                if exc.has_error_label('TransientTransactionError') and _within_time_limit(start_time):
                    break
                raise
            return ret