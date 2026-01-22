from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy.testing import in_
from ._autogen_fixtures import AutogenFixtureTest
from ... import testing
from ...testing import config
from ...testing import eq_
from ...testing import is_
from ...testing import TestBase
def _assert_alter_col(self, m1, m2, pk, nullable=None):
    ops = self._fixture(m1, m2, return_ops=True)
    modify_table = ops.ops[-1]
    alter_col = modify_table.ops[0]
    if nullable is None:
        eq_(alter_col.existing_nullable, not pk)
    else:
        eq_(alter_col.existing_nullable, nullable)
    assert alter_col.existing_type._compare_type_affinity(Integer())
    return alter_col