import collections
import contextlib
import copy
import fixtures
import pickle
import sys
from unittest import mock
import warnings
from oslo_config import cfg
from oslo_context import context as oslo_context
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy import String
from sqlalchemy import Table
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines as oslo_engines
from oslo_db.sqlalchemy import orm
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db import warning
class MockFacadeTest(test_base.BaseTestCase):
    """test by applying mocks to internal call-points.

    This applies mocks to
    oslo.db.sqlalchemy.engines.create_engine() and
    oslo.db.sqlalchemy.orm.get_maker(), then mocking a
    _TransactionFactory into
    oslo.db.sqlalchemy.enginefacade._context_manager._root_factory.

    Various scenarios are run against the enginefacade functions, and the
    exact calls made against the mock create_engine(), get_maker(), and
    associated objects are tested exactly against expected calls.

    """
    synchronous_reader = True
    engine_uri = 'some_connection'
    slave_uri = None

    def setUp(self):
        super(MockFacadeTest, self).setUp()
        writer_conn = SingletonConnection()
        writer_engine = SingletonEngine(writer_conn)
        writer_session = mock.Mock(connection=mock.Mock(return_value=writer_conn), info={})
        writer_maker = mock.Mock(return_value=writer_session)
        if self.slave_uri:
            async_reader_conn = SingletonConnection()
            async_reader_engine = SingletonEngine(async_reader_conn)
            async_reader_session = mock.Mock(connection=mock.Mock(return_value=async_reader_conn), info={})
            async_reader_maker = mock.Mock(return_value=async_reader_session)
        else:
            async_reader_conn = writer_conn
            async_reader_engine = writer_engine
            async_reader_session = writer_session
            async_reader_maker = writer_maker
        if self.synchronous_reader:
            reader_conn = writer_conn
            reader_engine = writer_engine
            reader_session = writer_session
            reader_maker = writer_maker
        else:
            reader_conn = async_reader_conn
            reader_engine = async_reader_engine
            reader_session = async_reader_session
            reader_maker = async_reader_maker
        self.connections = AssertDataSource(writer_conn, reader_conn, async_reader_conn)
        self.engines = AssertDataSource(writer_engine, reader_engine, async_reader_engine)
        self.sessions = AssertDataSource(writer_session, reader_session, async_reader_session)
        self.makers = AssertDataSource(writer_maker, reader_maker, async_reader_maker)

        def get_maker(engine, **kw):
            if engine is writer_engine:
                return self.makers.writer
            elif engine is reader_engine:
                return self.makers.reader
            elif engine is async_reader_engine:
                return self.makers.async_reader
            else:
                assert False
        session_patch = mock.patch.object(orm, 'get_maker', side_effect=get_maker)
        self.get_maker = session_patch.start()
        self.addCleanup(session_patch.stop)

        def create_engine(sql_connection, **kw):
            if sql_connection == self.engine_uri:
                return self.engines.writer
            elif sql_connection == self.slave_uri:
                return self.engines.async_reader
            else:
                assert False
        engine_patch = mock.patch.object(oslo_engines, 'create_engine', side_effect=create_engine)
        self.create_engine = engine_patch.start()
        self.addCleanup(engine_patch.stop)
        self.factory = enginefacade._TransactionFactory()
        self.factory.configure(synchronous_reader=self.synchronous_reader)
        self.factory.configure(connection=self.engine_uri, slave_connection=self.slave_uri)
        facade_patcher = mock.patch.object(enginefacade._context_manager, '_root_factory', self.factory)
        facade_patcher.start()
        self.addCleanup(facade_patcher.stop)

    def _assert_ctx_connection(self, context, connection):
        self.assertIs(context.connection, connection)

    def _assert_ctx_session(self, context, session):
        self.assertIs(context.session, session)

    def _assert_non_decorated_ctx_connection(self, context, connection):
        transaction_ctx = enginefacade._transaction_ctx_for_context(context)
        self.assertIs(transaction_ctx.connection, connection)

    def _assert_non_decorated_ctx_session(self, context, session):
        transaction_ctx = enginefacade._transaction_ctx_for_context(context)
        self.assertIs(transaction_ctx.session, session)

    @contextlib.contextmanager
    def _assert_engines(self):
        """produce a mock series of engine calls.

        These are expected to match engine-related calls established
        by the test subject.

        """
        writer_conn = SingletonConnection()
        writer_engine = SingletonEngine(writer_conn)
        if self.slave_uri:
            async_reader_conn = SingletonConnection()
            async_reader_engine = SingletonEngine(async_reader_conn)
        else:
            async_reader_conn = writer_conn
            async_reader_engine = writer_engine
        if self.synchronous_reader:
            reader_engine = writer_engine
        else:
            reader_engine = async_reader_engine
        engines = AssertDataSource(writer_engine, reader_engine, async_reader_engine)

        def create_engine(sql_connection, **kw):
            if sql_connection == self.engine_uri:
                return engines.writer
            elif sql_connection == self.slave_uri:
                return engines.async_reader
            else:
                assert False
        engine_factory = mock.Mock(side_effect=create_engine)
        engine_factory(sql_connection=self.engine_uri, **{k: mock.ANY for k in self.factory._engine_cfg.keys()})
        if self.slave_uri:
            engine_factory(sql_connection=self.slave_uri, **{k: mock.ANY for k in self.factory._engine_cfg.keys()})
        yield AssertDataSource(writer_engine, reader_engine, async_reader_engine)
        self.assertEqual(engine_factory.mock_calls, self.create_engine.mock_calls)
        for sym in [enginefacade._WRITER, enginefacade._READER, enginefacade._ASYNC_READER]:
            self.assertEqual(engines.element_for_writer(sym).mock_calls, self.engines.element_for_writer(sym).mock_calls)

    def _assert_async_reader_connection(self, engines, session=None):
        return self._assert_connection(engines, enginefacade._ASYNC_READER, session)

    def _assert_reader_connection(self, engines, session=None):
        return self._assert_connection(engines, enginefacade._READER, session)

    def _assert_writer_connection(self, engines, session=None):
        return self._assert_connection(engines, enginefacade._WRITER, session)

    @contextlib.contextmanager
    def _assert_connection(self, engines, writer, session=None):
        """produce a mock series of connection calls.

        These are expected to match connection-related calls established
        by the test subject.

        """
        if session:
            connection = session.connection()
            yield connection
        else:
            connection = engines.element_for_writer(writer).connect()
            trans = connection.begin()
            yield connection
            if writer is enginefacade._WRITER:
                trans.commit()
            else:
                trans.rollback()
            connection.close()
        self.assertEqual(connection.mock_calls, self.connections.element_for_writer(writer).mock_calls)

    @contextlib.contextmanager
    def _assert_makers(self, engines):
        writer_session = mock.Mock(connection=mock.Mock(return_value=engines.writer._assert_connection))
        writer_maker = mock.Mock(return_value=writer_session)
        if self.slave_uri:
            async_reader_session = mock.Mock(connection=mock.Mock(return_value=engines.async_reader._assert_connection))
            async_reader_maker = mock.Mock(return_value=async_reader_session)
        else:
            async_reader_session = writer_session
            async_reader_maker = writer_maker
        if self.synchronous_reader:
            reader_maker = writer_maker
        else:
            reader_maker = async_reader_maker
        makers = AssertDataSource(writer_maker, reader_maker, async_reader_maker)

        def get_maker(engine, **kw):
            if engine is engines.writer:
                return makers.writer
            elif engine is engines.reader:
                return makers.reader
            elif engine is engines.async_reader:
                return makers.async_reader
            else:
                assert False
        maker_factories = mock.Mock(side_effect=get_maker)
        maker_factories(engine=engines.writer, expire_on_commit=False)
        if self.slave_uri:
            maker_factories(engine=engines.async_reader, expire_on_commit=False)
        yield makers
        self.assertEqual(maker_factories.mock_calls, self.get_maker.mock_calls)
        for sym in [enginefacade._WRITER, enginefacade._READER, enginefacade._ASYNC_READER]:
            self.assertEqual(makers.element_for_writer(sym).mock_calls, self.makers.element_for_writer(sym).mock_calls)

    def _assert_async_reader_session(self, makers, connection=None, assert_calls=True):
        return self._assert_session(makers, enginefacade._ASYNC_READER, connection, assert_calls)

    def _assert_reader_session(self, makers, connection=None, assert_calls=True):
        return self._assert_session(makers, enginefacade._READER, connection, assert_calls)

    def _assert_writer_session(self, makers, connection=None, assert_calls=True):
        return self._assert_session(makers, enginefacade._WRITER, connection, assert_calls)

    def _emit_sub_writer_session(self, session):
        return self._emit_sub_session(enginefacade._WRITER, session)

    def _emit_sub_reader_session(self, session):
        return self._emit_sub_session(enginefacade._READER, session)

    @contextlib.contextmanager
    def _assert_session(self, makers, writer, connection=None, assert_calls=True):
        """produce a mock series of session calls.

        These are expected to match session-related calls established
        by the test subject.

        """
        if connection:
            session = makers.element_for_writer(writer)(bind=connection)
        else:
            session = makers.element_for_writer(writer)()
        session.begin()
        yield session
        if writer is enginefacade._WRITER:
            session.commit()
        elif enginefacade._context_manager._factory._transaction_ctx_cfg['rollback_reader_sessions']:
            session.rollback()
        session.close()
        if assert_calls:
            self.assertEqual(session.mock_calls, self.sessions.element_for_writer(writer).mock_calls)

    @contextlib.contextmanager
    def _emit_sub_session(self, writer, session):
        yield session
        if enginefacade._context_manager._factory._transaction_ctx_cfg['flush_on_subtransaction']:
            session.flush()

    def test_dispose_pool(self):
        facade = enginefacade.transaction_context()
        facade.configure(connection=self.engine_uri)
        facade.dispose_pool()
        self.assertFalse(hasattr(facade._factory, '_writer_engine'))
        facade._factory._start()
        facade.dispose_pool()
        self.assertEqual(facade._factory._writer_engine.pool.mock_calls, [mock.call.dispose()])

    def test_dispose_pool_w_reader(self):
        facade = enginefacade.transaction_context()
        facade.configure(connection=self.engine_uri, slave_connection=self.slave_uri)
        facade.dispose_pool()
        self.assertFalse(hasattr(facade._factory, '_writer_engine'))
        self.assertFalse(hasattr(facade._factory, '_reader_engine'))
        facade._factory._start()
        facade.dispose_pool()
        self.assertEqual(facade._factory._writer_engine.pool.mock_calls, [mock.call.dispose()])
        self.assertEqual(facade._factory._reader_engine.pool.mock_calls, [mock.call.dispose()])

    def test_started_flag(self):
        facade = enginefacade.transaction_context()
        self.assertFalse(facade.is_started)
        facade.configure(connection=self.engine_uri)
        facade.writer.get_engine()
        self.assertTrue(facade.is_started)

    def test_started_exception(self):
        facade = enginefacade.transaction_context()
        self.assertFalse(facade.is_started)
        facade.configure(connection=self.engine_uri)
        facade.writer.get_engine()
        exc = self.assertRaises(enginefacade.AlreadyStartedError, facade.configure, connection=self.engine_uri)
        self.assertEqual('this TransactionFactory is already started', exc.args[0])

    def test_session_reader_decorator(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader
        def go(context):
            context.session.execute('test')
        go(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers) as session:
                    session.execute('test')

    def test_session_reader_decorator_kwarg_call(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader
        def go(context):
            context.session.execute('test')
        go(context=context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers) as session:
                    session.execute('test')

    def test_connection_reader_decorator(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader.connection
        def go(context):
            context.connection.execute('test')
        go(context)
        with self._assert_engines() as engines:
            with self._assert_reader_connection(engines) as connection:
                connection.execute('test')

    def test_session_reader_nested_in_connection_reader(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader.connection
        def go1(context):
            context.connection.execute('test1')
            go2(context)

        @enginefacade.reader
        def go2(context):
            context.session.execute('test2')
        go1(context)
        with self._assert_engines() as engines:
            with self._assert_reader_connection(engines) as connection:
                connection.execute('test1')
                with self._assert_makers(engines) as makers:
                    with self._assert_reader_session(makers, connection) as session:
                        session.execute('test2')

    def test_connection_reader_nested_in_session_reader(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader
        def go1(context):
            context.session.execute('test1')
            go2(context)

        @enginefacade.reader.connection
        def go2(context):
            context.connection.execute('test2')
        go1(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers) as session:
                    session.execute('test1')
                    with self._assert_reader_connection(engines, session) as connection:
                        connection.execute('test2')

    def test_session_reader_decorator_nested(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader
        def go1(context):
            context.session.execute('test1')
            go2(context)

        @enginefacade.reader
        def go2(context):
            context.session.execute('test2')
        go1(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers) as session:
                    session.execute('test1')
                    session.execute('test2')

    def test_reader_nested_in_writer_ok(self):
        context = oslo_context.RequestContext()

        @enginefacade.writer
        def go1(context):
            context.session.execute('test1')
            go2(context)

        @enginefacade.reader
        def go2(context):
            context.session.execute('test2')
        go1(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_writer_session(makers) as session:
                    session.execute('test1')
                    session.execute('test2')

    def test_writer_nested_in_reader_raises(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader
        def go1(context):
            context.session.execute('test1')
            go2(context)

        @enginefacade.writer
        def go2(context):
            context.session.execute('test2')
        exc = self.assertRaises(TypeError, go1, context)
        self.assertEqual("Can't upgrade a READER transaction to a WRITER mid-transaction", exc.args[0])

    def test_async_on_writer_raises(self):
        exc = self.assertRaises(TypeError, getattr, enginefacade.writer, 'async_')
        self.assertEqual('Setting async on a WRITER makes no sense', exc.args[0])

    def test_savepoint_and_independent_raises(self):
        exc = self.assertRaises(TypeError, getattr, enginefacade.writer.independent, 'savepoint')
        self.assertEqual('setting savepoint and independent makes no sense.', exc.args[0])

    def test_reader_nested_in_async_reader_raises(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader.async_
        def go1(context):
            context.session.execute('test1')
            go2(context)

        @enginefacade.reader
        def go2(context):
            context.session.execute('test2')
        exc = self.assertRaises(TypeError, go1, context)
        self.assertEqual("Can't upgrade an ASYNC_READER transaction to a READER mid-transaction", exc.args[0])

    def test_reader_allow_async_nested_in_async_reader(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader.async_
        def go1(context):
            context.session.execute('test1')
            go2(context)

        @enginefacade.reader.allow_async
        def go2(context):
            context.session.execute('test2')
        go1(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_async_reader_session(makers) as session:
                    session.execute('test1')
                    session.execute('test2')

    def test_reader_allow_async_nested_in_reader(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader.reader
        def go1(context):
            context.session.execute('test1')
            go2(context)

        @enginefacade.reader.allow_async
        def go2(context):
            context.session.execute('test2')
        go1(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers) as session:
                    session.execute('test1')
                    session.execute('test2')

    def test_reader_allow_async_is_reader_by_default(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader.allow_async
        def go1(context):
            context.session.execute('test1')
        go1(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers) as session:
                    session.execute('test1')

    def test_writer_nested_in_async_reader_raises(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader.async_
        def go1(context):
            context.session.execute('test1')
            go2(context)

        @enginefacade.writer
        def go2(context):
            context.session.execute('test2')
        exc = self.assertRaises(TypeError, go1, context)
        self.assertEqual("Can't upgrade an ASYNC_READER transaction to a WRITER mid-transaction", exc.args[0])

    def test_reader_then_writer_ok(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader
        def go1(context):
            context.session.execute('test1')

        @enginefacade.writer
        def go2(context):
            context.session.execute('test2')
        go1(context)
        go2(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers, assert_calls=False) as session:
                    session.execute('test1')
                with self._assert_writer_session(makers) as session:
                    session.execute('test2')

    def test_deprecated_async_reader_name(self):
        if sys.version_info >= (3, 7):
            self.skipTest('Test only runs on Python < 3.7')
        context = oslo_context.RequestContext()
        old = getattr(enginefacade.reader, 'async')

        @old
        def go1(context):
            context.session.execute('test1')
        go1(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_async_reader_session(makers, assert_calls=False) as session:
                    session.execute('test1')

    def test_async_reader_then_reader_ok(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader.async_
        def go1(context):
            context.session.execute('test1')

        @enginefacade.reader
        def go2(context):
            context.session.execute('test2')
        go1(context)
        go2(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_async_reader_session(makers, assert_calls=False) as session:
                    session.execute('test1')
                with self._assert_reader_session(makers) as session:
                    session.execute('test2')

    def test_using_reader(self):
        context = oslo_context.RequestContext()
        with enginefacade.reader.using(context) as session:
            self._assert_ctx_session(context, session)
            session.execute('test1')
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers) as session:
                    session.execute('test1')

    def test_using_context_present_in_session_info(self):
        context = oslo_context.RequestContext()
        with enginefacade.reader.using(context) as session:
            self.assertEqual(context, session.info['using_context'])
        self.assertIsNone(session.info['using_context'])

    def test_using_context_present_in_connection_info(self):
        context = oslo_context.RequestContext()
        with enginefacade.writer.connection.using(context) as connection:
            self.assertEqual(context, connection.info['using_context'])
        self.assertIsNone(connection.info['using_context'])

    def test_using_reader_rollback_reader_session(self):
        enginefacade.configure(rollback_reader_sessions=True)
        context = oslo_context.RequestContext()
        with enginefacade.reader.using(context) as session:
            self._assert_ctx_session(context, session)
            session.execute('test1')
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers) as session:
                    session.execute('test1')

    def test_using_flush_on_nested(self):
        enginefacade.configure(flush_on_nested=True)
        context = oslo_context.RequestContext()
        with enginefacade.writer.using(context) as session:
            with enginefacade.writer.using(context) as session:
                self._assert_ctx_session(context, session)
                session.execute('test1')
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_writer_session(makers) as session:
                    with self._emit_sub_writer_session(session) as session:
                        session.execute('test1')

    def test_using_writer(self):
        context = oslo_context.RequestContext()
        with enginefacade.writer.using(context) as session:
            self._assert_ctx_session(context, session)
            session.execute('test1')
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_writer_session(makers) as session:
                    session.execute('test1')

    def test_using_writer_no_descriptors(self):
        context = NonDecoratedContext()
        with enginefacade.writer.using(context) as session:
            self._assert_non_decorated_ctx_session(context, session)
            session.execute('test1')
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_writer_session(makers) as session:
                    session.execute('test1')

    def test_using_writer_connection_no_descriptors(self):
        context = NonDecoratedContext()
        with enginefacade.writer.connection.using(context) as connection:
            self._assert_non_decorated_ctx_connection(context, connection)
            connection.execute('test1')
        with self._assert_engines() as engines:
            with self._assert_writer_connection(engines) as conn:
                conn.execute('test1')

    def test_using_reader_connection(self):
        context = oslo_context.RequestContext()
        with enginefacade.reader.connection.using(context) as connection:
            self._assert_ctx_connection(context, connection)
            connection.execute('test1')
        with self._assert_engines() as engines:
            with self._assert_reader_connection(engines) as conn:
                conn.execute('test1')

    def test_using_writer_connection(self):
        context = oslo_context.RequestContext()
        with enginefacade.writer.connection.using(context) as connection:
            self._assert_ctx_connection(context, connection)
            connection.execute('test1')
        with self._assert_engines() as engines:
            with self._assert_writer_connection(engines) as conn:
                conn.execute('test1')

    def test_context_copied_using_existing_writer_connection(self):
        context = oslo_context.RequestContext()
        with enginefacade.writer.connection.using(context) as connection:
            self._assert_ctx_connection(context, connection)
            connection.execute('test1')
            ctx2 = copy.deepcopy(context)
            with enginefacade.reader.connection.using(ctx2) as conn2:
                self.assertIs(conn2, connection)
                self._assert_ctx_connection(ctx2, conn2)
                conn2.execute('test2')
        with self._assert_engines() as engines:
            with self._assert_writer_connection(engines) as conn:
                conn.execute('test1')
                conn.execute('test2')

    def test_context_nodesc_copied_using_existing_writer_connection(self):
        context = NonDecoratedContext()
        with enginefacade.writer.connection.using(context) as connection:
            self._assert_non_decorated_ctx_connection(context, connection)
            connection.execute('test1')
            ctx2 = copy.deepcopy(context)
            with enginefacade.reader.connection.using(ctx2) as conn2:
                self.assertIs(conn2, connection)
                self._assert_non_decorated_ctx_connection(ctx2, conn2)
                conn2.execute('test2')
        with self._assert_engines() as engines:
            with self._assert_writer_connection(engines) as conn:
                conn.execute('test1')
                conn.execute('test2')

    def test_session_context_notrequested_exception(self):
        context = oslo_context.RequestContext()
        with enginefacade.reader.connection.using(context):
            exc = self.assertRaises(exception.ContextNotRequestedError, getattr, context, 'session')
            self.assertRegex(exc.args[0], "The 'session' context attribute was requested but it has not been established for this context.")

    def test_connection_context_notrequested_exception(self):
        context = oslo_context.RequestContext()
        with enginefacade.reader.using(context):
            exc = self.assertRaises(exception.ContextNotRequestedError, getattr, context, 'connection')
            self.assertRegex(exc.args[0], "The 'connection' context attribute was requested but it has not been established for this context.")

    def test_session_context_exception(self):
        context = oslo_context.RequestContext()
        exc = self.assertRaises(exception.NoEngineContextEstablished, getattr, context, 'session')
        self.assertRegex(exc.args[0], "No TransactionContext is established for this .*RequestContext.* object within the current thread; the 'session' attribute is unavailable.")

    def test_session_context_getattr(self):
        context = oslo_context.RequestContext()
        self.assertIsNone(getattr(context, 'session', None))

    def test_connection_context_exception(self):
        context = oslo_context.RequestContext()
        exc = self.assertRaises(exception.NoEngineContextEstablished, getattr, context, 'connection')
        self.assertRegex(exc.args[0], "No TransactionContext is established for this .*RequestContext.* object within the current thread; the 'connection' attribute is unavailable.")

    def test_connection_context_getattr(self):
        context = oslo_context.RequestContext()
        self.assertIsNone(getattr(context, 'connection', None))

    def test_transaction_context_exception(self):
        context = oslo_context.RequestContext()
        exc = self.assertRaises(exception.NoEngineContextEstablished, getattr, context, 'transaction')
        self.assertRegex(exc.args[0], "No TransactionContext is established for this .*RequestContext.* object within the current thread; the 'transaction' attribute is unavailable.")

    def test_transaction_context_getattr(self):
        context = oslo_context.RequestContext()
        self.assertIsNone(getattr(context, 'transaction', None))

    def test_trans_ctx_context_exception(self):
        context = oslo_context.RequestContext()
        exc = self.assertRaises(exception.NoEngineContextEstablished, getattr, context, 'transaction_ctx')
        self.assertRegex(exc.args[0], 'No TransactionContext is established for this .*RequestContext.* object within the current thread.')

    def test_trans_ctx_context_getattr(self):
        context = oslo_context.RequestContext()
        self.assertIsNone(getattr(context, 'transaction_ctx', None))

    def test_multiple_factories(self):
        """Test that the instrumentation applied to a context class is

        independent of a specific _TransactionContextManager
        / _TransactionFactory.

        """
        mgr1 = enginefacade.transaction_context()
        mgr1.configure(connection=self.engine_uri, slave_connection=self.slave_uri)
        mgr2 = enginefacade.transaction_context()
        mgr2.configure(connection=self.engine_uri, slave_connection=self.slave_uri)
        context = oslo_context.RequestContext()
        self.assertRaises(exception.NoEngineContextEstablished, getattr, context, 'session')
        with mgr1.writer.using(context):
            self.assertIs(context.transaction_ctx.factory, mgr1._factory)
            self.assertIsNot(context.transaction_ctx.factory, mgr2._factory)
            self.assertIsNotNone(context.session)
        self.assertRaises(exception.NoEngineContextEstablished, getattr, context, 'session')
        with mgr2.writer.using(context):
            self.assertIsNot(context.transaction_ctx.factory, mgr1._factory)
            self.assertIs(context.transaction_ctx.factory, mgr2._factory)
            self.assertIsNotNone(context.session)

    def test_multiple_factories_nested(self):
        """Test that the instrumentation applied to a context class supports

        nested calls among multiple _TransactionContextManager objects.

        """
        mgr1 = enginefacade.transaction_context()
        mgr1.configure(connection=self.engine_uri, slave_connection=self.slave_uri)
        mgr2 = enginefacade.transaction_context()
        mgr2.configure(connection=self.engine_uri, slave_connection=self.slave_uri)
        context = oslo_context.RequestContext()
        with mgr1.writer.using(context):
            self.assertIs(context.transaction_ctx.factory, mgr1._factory)
            self.assertIsNot(context.transaction_ctx.factory, mgr2._factory)
            with mgr2.writer.using(context):
                self.assertIsNot(context.transaction_ctx.factory, mgr1._factory)
                self.assertIs(context.transaction_ctx.factory, mgr2._factory)
                self.assertIsNotNone(context.session)
            self.assertIs(context.transaction_ctx.factory, mgr1._factory)
            self.assertIsNot(context.transaction_ctx.factory, mgr2._factory)
            self.assertIsNotNone(context.session)
        self.assertRaises(exception.NoEngineContextEstablished, getattr, context, 'transaction_ctx')

    def test_context_found_for_bound_method(self):
        context = oslo_context.RequestContext()

        @enginefacade.reader
        def go(self, context):
            context.session.execute('test')
        go(self, context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers) as session:
                    session.execute('test')

    def test_context_found_for_class_method(self):
        context = oslo_context.RequestContext()

        class Spam(object):

            @classmethod
            @enginefacade.reader
            def go(cls, context):
                context.session.execute('test')
        Spam.go(context)
        with self._assert_engines() as engines:
            with self._assert_makers(engines) as makers:
                with self._assert_reader_session(makers) as session:
                    session.execute('test')