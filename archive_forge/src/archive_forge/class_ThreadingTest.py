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
class ThreadingTest(db_test_base._DbTestCase):
    """Test copy/pickle on new threads using real connections and sessions."""

    def _assert_ctx_connection(self, context, connection):
        self.assertIs(context.connection, connection)

    def _assert_ctx_session(self, context, session):
        self.assertIs(context.session, session)

    def _patch_thread_ident(self):
        self.ident = 1
        test_instance = self

        class MockThreadingLocal(object):

            def __init__(self):
                self.__dict__['state'] = collections.defaultdict(dict)

            def __deepcopy__(self, memo):
                return self

            def __getattr__(self, key):
                ns = self.state[test_instance.ident]
                try:
                    return ns[key]
                except KeyError:
                    raise AttributeError(key)

            def __setattr__(self, key, value):
                ns = self.state[test_instance.ident]
                ns[key] = value

            def __delattr__(self, key):
                ns = self.state[test_instance.ident]
                try:
                    del ns[key]
                except KeyError:
                    raise AttributeError(key)
        return mock.patch.object(enginefacade, '_TransactionContextTLocal', MockThreadingLocal)

    def test_thread_ctxmanager_writer(self):
        context = oslo_context.RequestContext()
        with self._patch_thread_ident():
            with enginefacade.writer.using(context) as session:
                self._assert_ctx_session(context, session)
                self.ident = 2
                with enginefacade.reader.using(context) as sess2:
                    self.assertIsNot(sess2, session)
                    self._assert_ctx_session(context, sess2)
                self.ident = 1
                with enginefacade.reader.using(context) as sess3:
                    self.assertIs(sess3, session)
                    self._assert_ctx_session(context, session)

    def test_thread_ctxmanager_writer_connection(self):
        context = oslo_context.RequestContext()
        with self._patch_thread_ident():
            with enginefacade.writer.connection.using(context) as conn:
                self._assert_ctx_connection(context, conn)
                self.ident = 2
                with enginefacade.reader.connection.using(context) as conn2:
                    self.assertIsNot(conn2, conn)
                    self._assert_ctx_connection(context, conn2)
                    with enginefacade.reader.connection.using(context) as conn3:
                        self.assertIsNot(conn3, conn)
                        self.assertIs(conn3, conn2)
                self.ident = 1
                with enginefacade.reader.connection.using(context) as conn3:
                    self.assertIs(conn3, conn)
                    self._assert_ctx_connection(context, conn)

    def test_thread_ctxmanager_switch_styles(self):

        @enginefacade.writer.connection
        def go_one(context):
            self.assertRaises(exception.ContextNotRequestedError, getattr, context, 'session')
            self.assertIsNotNone(context.connection)
            self.ident = 2
            go_two(context)
            self.ident = 1
            self.assertRaises(exception.ContextNotRequestedError, getattr, context, 'session')
            self.assertIsNotNone(context.connection)

        @enginefacade.reader
        def go_two(context):
            self.assertRaises(exception.ContextNotRequestedError, getattr, context, 'connection')
            self.assertIsNotNone(context.session)
        context = oslo_context.RequestContext()
        with self._patch_thread_ident():
            go_one(context)

    def test_thread_decorator_writer(self):
        sessions = set()

        @enginefacade.writer
        def go_one(context):
            sessions.add(context.session)
            self.ident = 2
            go_two(context)
            self.ident = 1
            go_three(context)

        @enginefacade.reader
        def go_two(context):
            assert context.session not in sessions

        @enginefacade.reader
        def go_three(context):
            assert context.session in sessions
        context = oslo_context.RequestContext()
        with self._patch_thread_ident():
            go_one(context)

    def test_thread_decorator_writer_connection(self):
        connections = set()

        @enginefacade.writer.connection
        def go_one(context):
            connections.add(context.connection)
            self.ident = 2
            go_two(context)
            self.ident = 1
            go_three(context)

        @enginefacade.reader.connection
        def go_two(context):
            assert context.connection not in connections

        @enginefacade.reader
        def go_three(context):
            assert context.connection in connections
        context = oslo_context.RequestContext()
        with self._patch_thread_ident():
            go_one(context)

    def test_contexts_picklable(self):
        context = oslo_context.RequestContext()
        with enginefacade.writer.using(context) as session:
            self._assert_ctx_session(context, session)
            pickled = pickle.dumps(context)
            unpickled = pickle.loads(pickled)
            with enginefacade.writer.using(unpickled) as session2:
                self._assert_ctx_session(unpickled, session2)
                assert session is not session2