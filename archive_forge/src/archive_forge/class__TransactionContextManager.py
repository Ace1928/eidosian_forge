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
class _TransactionContextManager(object):
    """Provide context-management and decorator patterns for transactions.

    This object integrates user-defined "context" objects with the
    :class:`._TransactionContext` class, on behalf of a
    contained :class:`._TransactionFactory`.

    """

    def __init__(self, root=None, mode=None, independent=False, savepoint=False, connection=False, replace_global_factory=None, _is_global_manager=False, allow_async=False):
        if root is None:
            self._root = self
            self._root_factory = _TransactionFactory()
        else:
            self._root = root
        self._replace_global_factory = replace_global_factory
        self._is_global_manager = _is_global_manager
        self._mode = mode
        self._independent = independent
        self._savepoint = savepoint
        if self._savepoint and self._independent:
            raise TypeError('setting savepoint and independent makes no sense.')
        self._connection = connection
        self._allow_async = allow_async

    @property
    def _factory(self):
        """The :class:`._TransactionFactory` associated with this context."""
        return self._root._root_factory

    @property
    def is_started(self):
        """True if this manager is already started."""
        return self._factory.is_started

    def configure(self, **kw):
        """Apply configurational options to the factory.

        This method can only be called before any specific
        transaction-beginning methods have been called.


        """
        self._factory.configure(**kw)

    def append_on_engine_create(self, fn):
        """Append a listener function to _facade_cfg["on_engine_create"]"""
        self._factory._facade_cfg['on_engine_create'].append(fn)

    def get_legacy_facade(self):
        """Return a :class:`.LegacyEngineFacade` for factory from this context.

        This facade will make use of the same engine and sessionmaker
        as this factory, however will not share the same transaction context;
        the legacy facade continues to work the old way of returning
        a new Session each time get_session() is called.
        """
        return self._factory.get_legacy_facade()

    def get_engine(self):
        """Return the Engine in use.

        This will be based on the state being WRITER or READER.

        This implies a start operation.

        """
        if self._mode is _WRITER:
            return self._factory.get_writer_engine()
        elif self._mode is _READER:
            return self._factory.get_reader_engine()
        else:
            raise ValueError('mode should be WRITER or READER')

    def get_sessionmaker(self):
        """Return the sessionmaker in use.

        This will be based on the state being WRITER or READER.

        This implies a start operation.

        """
        if self._mode is _WRITER:
            return self._factory.get_writer_maker()
        elif self._mode is _READER:
            return self._factory.get_reader_maker()
        else:
            raise ValueError('mode should be WRITER or READER')

    def dispose_pool(self):
        """Call engine.pool.dispose() on underlying Engine objects."""
        self._factory.dispose_pool()

    def make_new_manager(self):
        """Create a new, independent _TransactionContextManager from this one.

        Copies the underlying _TransactionFactory to a new one, so that
        it can be further configured with new options.

        Used for test environments where the application-wide
        _TransactionContextManager may be used as a factory for test-local
        managers.

        """
        new = self._clone()
        new._root = new
        new._root_factory = self._root_factory._create_factory_copy()
        if new._factory._started:
            raise AssertionError('TransactionFactory is already started')
        return new

    def patch_factory(self, factory_or_manager):
        """Patch a _TransactionFactory into this manager.

        Replaces this manager's factory with the given one, and returns
        a callable that will reset the factory back to what we
        started with.

        Only works for root factories.  Is intended for test suites
        that need to patch in alternate database configurations.

        The given argument may be a _TransactionContextManager or a
        _TransactionFactory.

        """
        if isinstance(factory_or_manager, _TransactionContextManager):
            factory = factory_or_manager._factory
        elif isinstance(factory_or_manager, _TransactionFactory):
            factory = factory_or_manager
        else:
            raise ValueError('_TransactionContextManager or _TransactionFactory expected.')
        if self._root is not self:
            raise AssertionError('patch_factory only works for root factory.')
        existing_factory = self._root_factory
        self._root_factory = factory

        def reset():
            self._root_factory = existing_factory
        return reset

    def patch_engine(self, engine):
        """Patch an Engine into this manager.

        Replaces this manager's factory with a _TestTransactionFactory
        that will use the given Engine, and returns
        a callable that will reset the factory back to what we
        started with.

        Only works for root factories.  Is intended for test suites
        that need to patch in alternate database configurations.

        """
        existing_factory = self._factory
        if not existing_factory._started:
            existing_factory._start()
        maker = existing_factory._writer_maker
        maker_kwargs = existing_factory._maker_args_for_conf(cfg.CONF)
        maker = orm.get_maker(engine=engine, **maker_kwargs)
        factory = _TestTransactionFactory(engine, maker, apply_global=False, from_factory=existing_factory)
        return self.patch_factory(factory)

    @property
    def replace(self):
        """Modifier to replace the global transaction factory with this one."""
        return self._clone(replace_global_factory=self._factory)

    @property
    def writer(self):
        """Modifier to set the transaction to WRITER."""
        return self._clone(mode=_WRITER)

    @property
    def reader(self):
        """Modifier to set the transaction to READER."""
        return self._clone(mode=_READER)

    @property
    def allow_async(self):
        """Modifier to allow async operations

        Allows async operations if asynchronous session is already
        started in this context. Marking DB API methods with READER would make
        it impossible to use them in ASYNC_READER transactions, and marking
        them with ASYNC_READER would require a modification of all the places
        these DB API methods are called to force READER mode, where the latest
        DB state is required.

        In Nova DB API methods should have a 'safe' default (i.e. READER),
        so that they can start sessions on their own, but it would also be
        useful for them to be able to participate in an existing ASYNC_READER
        session, if one was started up the stack.
        """
        if self._mode is _WRITER:
            raise TypeError('Setting async on a WRITER makes no sense')
        return self._clone(allow_async=True)

    @property
    def independent(self):
        """Modifier to start a transaction independent from any enclosing."""
        return self._clone(independent=True)

    @property
    def savepoint(self):
        """Modifier to start a SAVEPOINT if a transaction already exists."""
        return self._clone(savepoint=True)

    @property
    def connection(self):
        """Modifier to return a core Connection object instead of Session."""
        return self._clone(connection=True)

    @property
    def async_(self):
        """Modifier to set a READER operation to ASYNC_READER."""
        if self._mode is _WRITER:
            raise TypeError('Setting async on a WRITER makes no sense')
        return self._clone(mode=_ASYNC_READER)

    def using(self, context):
        """Provide a context manager block that will use the given context."""
        return self._transaction_scope(context)

    def __call__(self, fn):
        """Decorate a function."""
        argspec = inspect.getfullargspec(fn)
        if argspec.args[0] == 'self' or argspec.args[0] == 'cls':
            context_index = 1
        else:
            context_index = 0
        context_kw = argspec.args[context_index]

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            context = kwargs.get(context_kw, None)
            if not context:
                context = args[context_index]
            with self._transaction_scope(context):
                return fn(*args, **kwargs)
        return wrapper

    def _clone(self, **kw):
        default_kw = {'independent': self._independent, 'mode': self._mode, 'connection': self._connection}
        default_kw.update(kw)
        return _TransactionContextManager(root=self._root, **default_kw)

    @contextlib.contextmanager
    def _transaction_scope(self, context):
        new_transaction = self._independent
        transaction_contexts_by_thread = _transaction_contexts_by_thread(context)
        current = restore = getattr(transaction_contexts_by_thread, 'current', None)
        use_factory = self._factory
        global_factory = None
        if self._replace_global_factory:
            use_factory = global_factory = self._replace_global_factory
        elif current is not None and current.global_factory:
            global_factory = current.global_factory
            if self._root._is_global_manager:
                use_factory = global_factory
        if current is not None and (new_transaction or current.factory is not use_factory):
            current = None
        if current is None:
            current = transaction_contexts_by_thread.current = _TransactionContext(use_factory, global_factory=global_factory)
        try:
            if self._mode is not None:
                with current._produce_block(mode=self._mode, connection=self._connection, savepoint=self._savepoint, allow_async=self._allow_async, context=context) as resource:
                    yield resource
            else:
                yield
        finally:
            if restore is None:
                del transaction_contexts_by_thread.current
            elif current is not restore:
                transaction_contexts_by_thread.current = restore