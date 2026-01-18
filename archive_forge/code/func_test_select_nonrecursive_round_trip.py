from .. import fixtures
from ..assertions import eq_
from ..schema import Column
from ..schema import Table
from ... import ForeignKey
from ... import Integer
from ... import select
from ... import String
from ... import testing
def test_select_nonrecursive_round_trip(self, connection):
    some_table = self.tables.some_table
    cte = select(some_table).where(some_table.c.data.in_(['d2', 'd3', 'd4'])).cte('some_cte')
    result = connection.execute(select(cte.c.data).where(cte.c.data.in_(['d4', 'd5'])))
    eq_(result.fetchall(), [('d4',)])