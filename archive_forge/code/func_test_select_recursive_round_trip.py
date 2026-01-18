from .. import fixtures
from ..assertions import eq_
from ..schema import Column
from ..schema import Table
from ... import ForeignKey
from ... import Integer
from ... import select
from ... import String
from ... import testing
def test_select_recursive_round_trip(self, connection):
    some_table = self.tables.some_table
    cte = select(some_table).where(some_table.c.data.in_(['d2', 'd3', 'd4'])).cte('some_cte', recursive=True)
    cte_alias = cte.alias('c1')
    st1 = some_table.alias()
    cte = cte.union_all(select(st1).where(st1.c.id == cte_alias.c.parent_id))
    result = connection.execute(select(cte.c.data).where(cte.c.data != 'd2').order_by(cte.c.data.desc()))
    eq_(result.fetchall(), [('d4',), ('d3',), ('d3',), ('d1',), ('d1',), ('d1',)])