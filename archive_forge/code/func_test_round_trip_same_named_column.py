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
@tough_parameters
@config.requirements.unusual_column_name_characters
def test_round_trip_same_named_column(self, paramname, connection, metadata):
    name = paramname
    t = Table('t', metadata, Column('id', Integer, primary_key=True), Column(name, String(50), nullable=False))
    t.create(connection)
    connection.execute(t.insert().values({'id': 1, name: 'some name'}))
    stmt = select(t.c[name]).where(t.c[name] == 'some name')
    eq_(connection.scalar(stmt), 'some name')
    stmt = select(t.c[name]).where(t.c[name] == bindparam(name))
    row = connection.execute(stmt, {name: 'some name'}).first()
    eq_(row._mapping[name], 'some name')
    stmt = select(t.c[name]).where(t.c[name].in_(['some name', 'some other_name']))
    row = connection.execute(stmt).first()