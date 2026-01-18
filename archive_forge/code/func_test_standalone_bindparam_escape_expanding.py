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
def test_standalone_bindparam_escape_expanding(self, paramname, connection, multirow_fixture):
    tbl1 = multirow_fixture
    stmt = select(tbl1.c.myid).where(tbl1.c.name.in_(bindparam(paramname, value=['a', 'b']))).order_by(tbl1.c.myid)
    res = connection.scalars(stmt, {paramname: ['d', 'a']}).all()
    eq_(res, [1, 4])