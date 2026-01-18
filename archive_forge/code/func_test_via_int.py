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
def test_via_int(self, connection):
    row = connection.execute(self.tables.plain_pk.select().order_by(self.tables.plain_pk.c.id)).first()
    eq_(row[0], 1)
    eq_(row[1], 'd1')