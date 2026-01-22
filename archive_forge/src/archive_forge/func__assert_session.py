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