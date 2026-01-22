from __future__ import annotations
import configparser
from contextlib import contextmanager
import io
import re
from typing import Any
from typing import Dict
from sqlalchemy import Column
from sqlalchemy import inspect
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import testing
from sqlalchemy import text
from sqlalchemy.testing import config
from sqlalchemy.testing import mock
from sqlalchemy.testing.assertions import eq_
from sqlalchemy.testing.fixtures import TablesTest as SQLAlchemyTablesTest
from sqlalchemy.testing.fixtures import TestBase as SQLAlchemyTestBase
import alembic
from .assertions import _get_dialect
from ..environment import EnvironmentContext
from ..migration import MigrationContext
from ..operations import Operations
from ..util import sqla_compat
from ..util.sqla_compat import create_mock_engine
from ..util.sqla_compat import sqla_14
from ..util.sqla_compat import sqla_2
class AlterColRoundTripFixture:
    __requires__ = ('alter_column',)

    def setUp(self):
        self.conn = config.db.connect()
        self.ctx = MigrationContext.configure(self.conn)
        self.op = Operations(self.ctx)
        self.metadata = MetaData()

    def _compare_type(self, t1, t2):
        c1 = Column('q', t1)
        c2 = Column('q', t2)
        assert not self.ctx.impl.compare_type(c1, c2), "Type objects %r and %r didn't compare as equivalent" % (t1, t2)

    def _compare_server_default(self, t1, s1, t2, s2):
        c1 = Column('q', t1, server_default=s1)
        c2 = Column('q', t2, server_default=s2)
        assert not self.ctx.impl.compare_server_default(c1, c2, s2, s1), "server defaults %r and %r didn't compare as equivalent" % (s1, s2)

    def tearDown(self):
        sqla_compat._safe_rollback_connection_transaction(self.conn)
        with self.conn.begin():
            self.metadata.drop_all(self.conn)
        self.conn.close()

    def _run_alter_col(self, from_, to_, compare=None):
        column = Column(from_.get('name', 'colname'), from_.get('type', String(10)), nullable=from_.get('nullable', True), server_default=from_.get('server_default', None))
        t = Table('x', self.metadata, column)
        with sqla_compat._ensure_scope_for_ddl(self.conn):
            t.create(self.conn)
            insp = inspect(self.conn)
            old_col = insp.get_columns('x')[0]
            self.op.alter_column('x', column.name, existing_type=column.type, existing_server_default=column.server_default if column.server_default is not None else False, existing_nullable=True if column.nullable else False, nullable=to_.get('nullable', None), server_default=to_.get('server_default', False), new_column_name=to_.get('name', None), type_=to_.get('type', None))
        insp = inspect(self.conn)
        new_col = insp.get_columns('x')[0]
        if compare is None:
            compare = to_
        eq_(new_col['name'], compare['name'] if 'name' in compare else column.name)
        self._compare_type(new_col['type'], compare.get('type', old_col['type']))
        eq_(new_col['nullable'], compare.get('nullable', column.nullable))
        self._compare_server_default(new_col['type'], new_col.get('default', None), compare.get('type', old_col['type']), compare['server_default'].text if 'server_default' in compare else column.server_default.arg.text if column.server_default is not None else None)