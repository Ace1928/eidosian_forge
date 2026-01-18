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
@requirements.duplicate_names_in_cursor_description
def test_row_with_dupe_names(self, connection):
    result = connection.execute(select(self.tables.plain_pk.c.data, self.tables.plain_pk.c.data.label('data')).order_by(self.tables.plain_pk.c.id))
    row = result.first()
    eq_(result.keys(), ['data', 'data'])
    eq_(row, ('d1', 'd1'))