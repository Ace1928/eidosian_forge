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