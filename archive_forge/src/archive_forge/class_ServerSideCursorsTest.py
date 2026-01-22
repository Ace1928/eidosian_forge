import datetime
from .. import engines
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import DateTime
from ... import func
from ... import Integer
from ... import select
from ... import sql
from ... import String
from ... import testing
from ... import text
class ServerSideCursorsTest(fixtures.TestBase, testing.AssertsExecutionResults):
    __requires__ = ('server_side_cursors',)
    __backend__ = True

    def _is_server_side(self, cursor):
        if self.engine.dialect.driver == 'psycopg2':
            return bool(cursor.name)
        elif self.engine.dialect.driver == 'pymysql':
            sscursor = __import__('pymysql.cursors').cursors.SSCursor
            return isinstance(cursor, sscursor)
        elif self.engine.dialect.driver in ('aiomysql', 'asyncmy', 'aioodbc'):
            return cursor.server_side
        elif self.engine.dialect.driver == 'mysqldb':
            sscursor = __import__('MySQLdb.cursors').cursors.SSCursor
            return isinstance(cursor, sscursor)
        elif self.engine.dialect.driver == 'mariadbconnector':
            return not cursor.buffered
        elif self.engine.dialect.driver in ('asyncpg', 'aiosqlite'):
            return cursor.server_side
        elif self.engine.dialect.driver == 'pg8000':
            return getattr(cursor, 'server_side', False)
        elif self.engine.dialect.driver == 'psycopg':
            return bool(getattr(cursor, 'name', False))
        else:
            return False

    def _fixture(self, server_side_cursors):
        if server_side_cursors:
            with testing.expect_deprecated('The create_engine.server_side_cursors parameter is deprecated and will be removed in a future release.  Please use the Connection.execution_options.stream_results parameter.'):
                self.engine = engines.testing_engine(options={'server_side_cursors': server_side_cursors})
        else:
            self.engine = engines.testing_engine(options={'server_side_cursors': server_side_cursors})
        return self.engine

    @testing.combinations(('global_string', True, 'select 1', True), ('global_text', True, text('select 1'), True), ('global_expr', True, select(1), True), ('global_off_explicit', False, text('select 1'), False), ('stmt_option', False, select(1).execution_options(stream_results=True), True), ('stmt_option_disabled', True, select(1).execution_options(stream_results=False), False), ('for_update_expr', True, select(1).with_for_update(), True), ('for_update_string', True, 'SELECT 1 FOR UPDATE', True, testing.skip_if(['sqlite', 'mssql'])), ('text_no_ss', False, text('select 42'), False), ('text_ss_option', False, text('select 42').execution_options(stream_results=True), True), id_='iaaa', argnames='engine_ss_arg, statement, cursor_ss_status')
    def test_ss_cursor_status(self, engine_ss_arg, statement, cursor_ss_status):
        engine = self._fixture(engine_ss_arg)
        with engine.begin() as conn:
            if isinstance(statement, str):
                result = conn.exec_driver_sql(statement)
            else:
                result = conn.execute(statement)
            eq_(self._is_server_side(result.cursor), cursor_ss_status)
            result.close()

    def test_conn_option(self):
        engine = self._fixture(False)
        with engine.connect() as conn:
            result = conn.execution_options(stream_results=True).exec_driver_sql('select 1')
            assert self._is_server_side(result.cursor)
            result.close()

    def test_stmt_enabled_conn_option_disabled(self):
        engine = self._fixture(False)
        s = select(1).execution_options(stream_results=True)
        with engine.connect() as conn:
            result = conn.execution_options(stream_results=False).execute(s)
            assert not self._is_server_side(result.cursor)

    def test_aliases_and_ss(self):
        engine = self._fixture(False)
        s1 = select(sql.literal_column('1').label('x')).execution_options(stream_results=True).subquery()
        with engine.begin() as conn:
            result = conn.execute(s1.select())
            assert not self._is_server_side(result.cursor)
            result.close()
        s2 = select(1).select_from(s1)
        with engine.begin() as conn:
            result = conn.execute(s2)
            assert not self._is_server_side(result.cursor)
            result.close()

    def test_roundtrip_fetchall(self, metadata):
        md = self.metadata
        engine = self._fixture(True)
        test_table = Table('test_table', md, Column('id', Integer, primary_key=True), Column('data', String(50)))
        with engine.begin() as connection:
            test_table.create(connection, checkfirst=True)
            connection.execute(test_table.insert(), dict(data='data1'))
            connection.execute(test_table.insert(), dict(data='data2'))
            eq_(connection.execute(test_table.select().order_by(test_table.c.id)).fetchall(), [(1, 'data1'), (2, 'data2')])
            connection.execute(test_table.update().where(test_table.c.id == 2).values(data=test_table.c.data + ' updated'))
            eq_(connection.execute(test_table.select().order_by(test_table.c.id)).fetchall(), [(1, 'data1'), (2, 'data2 updated')])
            connection.execute(test_table.delete())
            eq_(connection.scalar(select(func.count('*')).select_from(test_table)), 0)

    def test_roundtrip_fetchmany(self, metadata):
        md = self.metadata
        engine = self._fixture(True)
        test_table = Table('test_table', md, Column('id', Integer, primary_key=True), Column('data', String(50)))
        with engine.begin() as connection:
            test_table.create(connection, checkfirst=True)
            connection.execute(test_table.insert(), [dict(data='data%d' % i) for i in range(1, 20)])
            result = connection.execute(test_table.select().order_by(test_table.c.id))
            eq_(result.fetchmany(5), [(i, 'data%d' % i) for i in range(1, 6)])
            eq_(result.fetchmany(10), [(i, 'data%d' % i) for i in range(6, 16)])
            eq_(result.fetchall(), [(i, 'data%d' % i) for i in range(16, 20)])