import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table
from alembic.util import sqla_compat
from ._autogen_fixtures import AutogenFixtureTest
from ... import testing
from ...testing import config
from ...testing import eq_
from ...testing import is_true
from ...testing import TestBase
def test_remove_identity_from_column(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('user', m1, Column('id', Integer, sa.Identity(start=2, maxvalue=1000)), Column('other', sa.Text))
    Table('user', m2, Column('id', Integer), Column('other', sa.Text))
    diffs = self._fixture(m1, m2)
    eq_(len(diffs[0]), 1)
    diffs = diffs[0][0]
    eq_(diffs[0], 'modify_default')
    eq_(diffs[2], 'user')
    eq_(diffs[3], 'id')
    eq_(diffs[6], None)
    removed = diffs[5]
    is_true(isinstance(removed, sa.Identity))