import importlib
from . import testing
from .. import assert_raises
from .. import config
from .. import engines
from .. import eq_
from .. import fixtures
from .. import is_not_none
from .. import is_true
from .. import ne_
from .. import provide_metadata
from ..assertions import expect_raises
from ..assertions import expect_raises_message
from ..config import requirements
from ..provision import set_default_schema_on_connection
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import dialects
from ... import event
from ... import exc
from ... import Integer
from ... import literal_column
from ... import select
from ... import String
from ...sql.compiler import Compiled
from ...util import inspect_getfullargspec
@provide_metadata
def test_percent_sign_round_trip(self):
    """test that the DBAPI accommodates for escaped / nonescaped
        percent signs in a way that matches the compiler

        """
    m = self.metadata
    t = Table('t', m, Column('data', String(50)))
    t.create(config.db)
    with config.db.begin() as conn:
        conn.execute(t.insert(), dict(data='some % value'))
        conn.execute(t.insert(), dict(data='some %% other value'))
        eq_(conn.scalar(select(t.c.data).where(t.c.data == literal_column("'some % value'"))), 'some % value')
        eq_(conn.scalar(select(t.c.data).where(t.c.data == literal_column("'some %% other value'"))), 'some %% other value')