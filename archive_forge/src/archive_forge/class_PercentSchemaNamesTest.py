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
class PercentSchemaNamesTest(fixtures.TablesTest):
    """tests using percent signs, spaces in table and column names.

    This didn't work for PostgreSQL / MySQL drivers for a long time
    but is now supported.

    """
    __requires__ = ('percent_schema_names',)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        cls.tables.percent_table = Table('percent%table', metadata, Column('percent%', Integer), Column('spaces % more spaces', Integer))
        cls.tables.lightweight_percent_table = sql.table('percent%table', sql.column('percent%'), sql.column('spaces % more spaces'))

    def test_single_roundtrip(self, connection):
        percent_table = self.tables.percent_table
        for params in [{'percent%': 5, 'spaces % more spaces': 12}, {'percent%': 7, 'spaces % more spaces': 11}, {'percent%': 9, 'spaces % more spaces': 10}, {'percent%': 11, 'spaces % more spaces': 9}]:
            connection.execute(percent_table.insert(), params)
        self._assert_table(connection)

    def test_executemany_roundtrip(self, connection):
        percent_table = self.tables.percent_table
        connection.execute(percent_table.insert(), {'percent%': 5, 'spaces % more spaces': 12})
        connection.execute(percent_table.insert(), [{'percent%': 7, 'spaces % more spaces': 11}, {'percent%': 9, 'spaces % more spaces': 10}, {'percent%': 11, 'spaces % more spaces': 9}])
        self._assert_table(connection)

    @requirements.insert_executemany_returning
    def test_executemany_returning_roundtrip(self, connection):
        percent_table = self.tables.percent_table
        connection.execute(percent_table.insert(), {'percent%': 5, 'spaces % more spaces': 12})
        result = connection.execute(percent_table.insert().returning(percent_table.c['percent%'], percent_table.c['spaces % more spaces']), [{'percent%': 7, 'spaces % more spaces': 11}, {'percent%': 9, 'spaces % more spaces': 10}, {'percent%': 11, 'spaces % more spaces': 9}])
        eq_(result.all(), [(7, 11), (9, 10), (11, 9)])
        self._assert_table(connection)

    def _assert_table(self, conn):
        percent_table = self.tables.percent_table
        lightweight_percent_table = self.tables.lightweight_percent_table
        for table in (percent_table, percent_table.alias(), lightweight_percent_table, lightweight_percent_table.alias()):
            eq_(list(conn.execute(table.select().order_by(table.c['percent%']))), [(5, 12), (7, 11), (9, 10), (11, 9)])
            eq_(list(conn.execute(table.select().where(table.c['spaces % more spaces'].in_([9, 10])).order_by(table.c['percent%']))), [(9, 10), (11, 9)])
            row = conn.execute(table.select().order_by(table.c['percent%'])).first()
            eq_(row._mapping['percent%'], 5)
            eq_(row._mapping['spaces % more spaces'], 12)
            eq_(row._mapping[table.c['percent%']], 5)
            eq_(row._mapping[table.c['spaces % more spaces']], 12)
        conn.execute(percent_table.update().values({percent_table.c['spaces % more spaces']: 15}))
        eq_(list(conn.execute(percent_table.select().order_by(percent_table.c['percent%']))), [(5, 15), (7, 15), (9, 15), (11, 15)])