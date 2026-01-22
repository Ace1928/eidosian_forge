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