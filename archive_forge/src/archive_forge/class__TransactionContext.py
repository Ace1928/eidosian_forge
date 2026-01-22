import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
class _TransactionContext(object):
    """Represent a single database transaction in progress."""

    def __init__(self, factory, global_factory=None):
        """Construct a new :class:`.TransactionContext`.

        :param factory: the :class:`.TransactionFactory` which will
            serve as a source of connectivity.
        :param global_factory: the "global" factory which will be used
            by the global ``_context_manager`` for new ``_TransactionContext``
            objects created under this one.  When left as None the actual
            "global" factory is used.
        """
        self.factory = factory
        self.global_factory = global_factory
        self.mode = None
        self.session = None
        self.connection = None
        self.transaction = None
        kw = self.factory._transaction_ctx_cfg
        self.rollback_reader_sessions = kw['rollback_reader_sessions']
        self.flush_on_subtransaction = kw['flush_on_subtransaction']

    @contextlib.contextmanager
    def _connection(self, savepoint=False, context=None):
        if self.connection is None:
            try:
                if self.session is not None:
                    self.connection = self.session.connection()
                    if savepoint:
                        with self.connection.begin_nested(), self._add_context(self.connection, context):
                            yield self.connection
                    else:
                        with self._add_context(self.connection, context):
                            yield self.connection
                else:
                    self.connection = self.factory._create_connection(mode=self.mode)
                    self.transaction = self.connection.begin()
                    try:
                        with self._add_context(self.connection, context):
                            yield self.connection
                        self._end_connection_transaction(self.transaction)
                    except Exception:
                        self.transaction.rollback()
                        raise
                    finally:
                        self.transaction = None
                        self.connection.close()
            finally:
                self.connection = None
        elif savepoint:
            with self.connection.begin_nested(), self._add_context(self.connection, context):
                yield self.connection
        else:
            with self._add_context(self.connection, context):
                yield self.connection

    @contextlib.contextmanager
    def _session(self, savepoint=False, context=None):
        if self.session is None:
            self.session = self.factory._create_session(bind=self.connection, mode=self.mode)
            try:
                self.session.begin()
                with self._add_context(self.session, context):
                    yield self.session
                self._end_session_transaction(self.session)
            except Exception:
                with excutils.save_and_reraise_exception():
                    self.session.rollback()
            finally:
                self.session.close()
                self.session = None
        elif savepoint:
            with self.session.begin_nested():
                with self._add_context(self.session, context):
                    yield self.session
        else:
            with self._add_context(self.session, context):
                yield self.session
            if self.flush_on_subtransaction:
                self.session.flush()

    @contextlib.contextmanager
    def _add_context(self, connection, context):
        restore_context = connection.info.get('using_context')
        connection.info['using_context'] = context
        yield connection
        connection.info['using_context'] = restore_context

    def _end_session_transaction(self, session):
        if self.mode is _WRITER:
            session.commit()
        elif self.rollback_reader_sessions:
            session.rollback()

    def _end_connection_transaction(self, transaction):
        if self.mode is _WRITER:
            transaction.commit()
        else:
            transaction.rollback()

    def _produce_block(self, mode, connection, savepoint, allow_async=False, context=None):
        if mode is _WRITER:
            self._writer()
        elif mode is _ASYNC_READER:
            self._async_reader()
        else:
            self._reader(allow_async)
        if connection:
            return self._connection(savepoint, context=context)
        else:
            return self._session(savepoint, context=context)

    def _writer(self):
        if self.mode is None:
            self.mode = _WRITER
        elif self.mode is _READER:
            raise TypeError("Can't upgrade a READER transaction to a WRITER mid-transaction")
        elif self.mode is _ASYNC_READER:
            raise TypeError("Can't upgrade an ASYNC_READER transaction to a WRITER mid-transaction")

    def _reader(self, allow_async=False):
        if self.mode is None:
            self.mode = _READER
        elif self.mode is _ASYNC_READER and (not allow_async):
            raise TypeError("Can't upgrade an ASYNC_READER transaction to a READER mid-transaction")

    def _async_reader(self):
        if self.mode is None:
            self.mode = _ASYNC_READER